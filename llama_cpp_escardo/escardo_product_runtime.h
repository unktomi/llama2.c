#ifndef LLAMA_CPP_ESCARDO_PRODUCT_RUNTIME_H
#define LLAMA_CPP_ESCARDO_PRODUCT_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace escardo_product {

using BindingId = std::uint64_t;
using NodeId = std::uint64_t;
using ObserverId = std::uint64_t;
using SelectionId = std::uint64_t;
using DemandId = std::uint64_t;
using Token = std::int32_t;

/* Borrowed references into backend-owned KV and hidden-state storage. */
struct KvSummaryRef {
    std::uint64_t handle = 0;
    std::int32_t position = -1;
};

struct HiddenStateRef {
    std::uint64_t handle = 0;
    const float * data = nullptr;
    std::size_t width = 0;
};

struct VocabularyCovectorRef {
    std::uint64_t handle = 0;
    const float * data = nullptr;
    std::size_t width = 0;
};

struct StructuredOutcomeRef {
    KvSummaryRef kv_summary;
    HiddenStateRef final_hidden;
    VocabularyCovectorRef proposal;
};

struct ProductNode;

/*
 * The constructor, its continuation term, and its structured outcome remain
 * one object. BindingId is the stable row identity through llama.cpp/Metal
 * batches; a backend must not split these into unkeyed parallel arrays.
 */
struct BoundContinuation {
    BindingId binding_id = 0;
    Token token = 0;
    std::int32_t local_rank = 0;
    std::size_t position = 0;
    StructuredOutcomeRef outcome;
    const ProductNode * continuation = nullptr;
};

/*
 * A node is a local selection function at one position. `history` is the
 * common causal boundary before its candidate x. `proposal` is the unit-law
 * observer for an empty suffix. final_hidden remains part of the structured
 * outcome, but is never fed through the model a second time by this runtime.
 */
struct ProductNode {
    NodeId node_id = 0;
    std::size_t position = 0;
    StructuredOutcomeRef history;
    std::vector<BoundContinuation> alternatives;
};

struct BoundPath {
    std::vector<const BoundContinuation *> positions;

    std::vector<Token> tokens() const;
};

struct ObservationFrame {
    ObserverId observer_id = 0;
    std::uint64_t frame_id = 0;

    bool operator==(const ObservationFrame & other) const;
    bool operator!=(const ObservationFrame & other) const;
};

/* A coordinate can only be ordered against another coordinate in its frame. */
struct FramedCoordinate {
    Token token = 0;
    ObservationFrame frame;
    double value = 0.0;
};

/* One projective/affine covector over a node's complete common support. */
struct PositionCovector {
    std::size_t position = 0;
    ObservationFrame frame;
    std::vector<FramedCoordinate> coordinates;
};

/* R in J_R X: a zip of covectors, never a fold or path score. */
struct ObservationTuple {
    std::vector<PositionCovector> positions;
};

/*
 * One restricted root-observer argument p(x : b(x)). `candidate` names x and
 * selected_suffix fixes b(x). For every x' in common_support, the backend
 * returns the coordinate
 *
 *     log P_model(x' ++ selected_suffix | fixed left history).
 *
 * All x' coordinates for this one fixed suffix inhabit one frame. Covectors
 * for two different suffixes are different frames and are never compared.
 */
struct ObservationDemand {
    DemandId demand_id = 0;
    const BoundContinuation * candidate = nullptr;
    StructuredOutcomeRef history_before_candidate;
    BoundPath selected_suffix;
    std::vector<Token> common_support;
};

/* Sibling restrictions may be lowered together in one llama.cpp batch. */
struct ObservationBatch {
    SelectionId selection_id = 0;
    std::size_t selecting_position = 0;
    std::vector<ObservationDemand> demands;
};

struct DemandObservation {
    DemandId demand_id = 0;
    PositionCovector covector;
};

struct ObservationBatchResult {
    SelectionId selection_id = 0;
    std::vector<DemandObservation> observations;
};

using PosteriorLaneId = std::uint64_t;

/*
 * Stable lowering identity for one coordinate of one demand. The GGUF
 * backend copies `history_before_candidate.kv_summary` to a lane sequence and
 * teacher-forces candidate_then_suffix. The coordinate is the history
 * proposal log-probability of candidate_token plus the conditional
 * log-probabilities of the fixed suffix tokens. This sum is internal to one
 * causal-posterior coordinate; it is not a score folded across Selects.
 *
 * For an empty suffix proposal_only is true: no decode is permitted, and the
 * coordinate comes directly from history_before_candidate.proposal.
 */
struct CausalPosteriorLane {
    PosteriorLaneId lane_id = 0;
    DemandId demand_id = 0;
    Token candidate_token = 0;
    StructuredOutcomeRef history_before_candidate;
    std::vector<Token> candidate_then_suffix;
    bool proposal_only = false;
};

struct CausalPosteriorSchedule {
    SelectionId selection_id = 0;
    std::size_t position = 0;
    std::vector<CausalPosteriorLane> lanes;
};

/* Expands demands without evaluating or rating them. */
CausalPosteriorSchedule make_causal_posterior_schedule(
    const ObservationBatch & batch
);

/* The same closure object receives every recursively restricted argument. */
class RootObserver {
public:
    virtual ~RootObserver() = default;

    virtual ObserverId observer_id() const = 0;
    virtual ObservationBatchResult observe(
        const ObservationBatch & batch
    ) = 0;
};

struct ObservedAlternative {
    const BoundContinuation * binding = nullptr;
    const BoundPath * selected_suffix = nullptr;
    const BoundPath * complete_path = nullptr;
    const PositionCovector * current_covector = nullptr;
    const ObservationTuple * observation_tuple = nullptr;
    bool attains = false;
};

class ProductEventSink {
public:
    virtual ~ProductEventSink() = default;

    virtual void observation_batch_requested(
        const ObservationBatch & /* batch */
    ) { }

    virtual void candidate_observed(
        SelectionId /* selection_id */,
        const ObservedAlternative & /* alternative */
    ) { }

    virtual void continuation_selected(
        SelectionId /* selection_id */,
        const ObservedAlternative & /* selected */
    ) { }
};

struct ProductCounters {
    std::uint64_t selection_nodes = 0;
    std::uint64_t observer_batches = 0;
    std::uint64_t observer_demands = 0;
};

struct ProductResult {
    BoundPath path;
    ObservationTuple observation;
};

/*
 * Mechanical finite dependent product:
 *
 *   b(x)   = product(delta(x), xs -> p(x : xs))
 *   a      = epsilon(x -> p(x : b(x)))
 *   result = a : b(a)
 *
 * For this finite-search epsilon, x attains its own observer result exactly
 * when x is an argmax coordinate of that result's support covector. The first
 * attaining x is selected; if none attains, the last support element is the
 * total fallback. Values from two different observer frames are never
 * compared. The selected current covector is prepended to b(a)'s retained
 * position-indexed tuple.
 */
class ExactProduct {
public:
    ExactProduct(
        std::size_t horizon,
        RootObserver & root_observer,
        ProductEventSink * events = nullptr
    );

    ProductResult run(const ProductNode & root);
    const ProductCounters & counters() const;

private:
    ProductResult evaluate(
        const ProductNode & node,
        std::vector<const ProductNode *> & active
    );

    std::vector<PositionCovector> observe_restrictions(
        SelectionId selection_id,
        const ProductNode & node,
        const std::vector<BoundPath> & selected_suffixes
    );

    std::size_t horizon_ = 0;
    RootObserver & root_observer_;
    ProductEventSink * events_ = nullptr;
    ProductCounters counters_;
    SelectionId next_selection_id_ = 1;
    DemandId next_demand_id_ = 1;
};

} // namespace escardo_product

#endif
