/*
 * QUARANTINED with the causal-posterior/boundary-aggregate scorer that this
 * runtime represents.  Its scheduling ideas may be inspected independently,
 * but the typed outcome and backup API encode a scorer that failed system
 * cases and must not be restored wholesale as an active inference contract.
 */

#ifndef LLAMA_CPP_ESCARDO_PRODUCT_RUNTIME_H
#define LLAMA_CPP_ESCARDO_PRODUCT_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <limits>
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

/*
 * A constructor and the state obtained by forcing that constructor remain one
 * object. BindingId is the stable row identity through llama.cpp/Metal
 * batches; a backend must not split these into unkeyed parallel arrays.
 *
 * A terminal binding deliberately has child_materialized == false: the
 * SelectionTerm need not retain that child as a recursive frontier. The root
 * observer still teacher-forces the terminal constructor in its scan lane so
 * the outgoing causal boundary participates in the affine summary.
 */
struct BoundContinuation {
    BindingId binding_id = 0;
    Token token = 0;
    std::int32_t local_rank = 0;
    std::size_t position = 0;
    float proposal_logit = 0.0f;
    double proposal_log_probability = 0.0;
    bool child_materialized = false;
    StructuredOutcomeRef outcome;
};

/* Stable, owning path data that survives release of a forced model frontier. */
struct BoundValue {
    BindingId binding_id = 0;
    Token token = 0;
    std::int32_t local_rank = 0;
    std::size_t position = 0;
    float proposal_logit = 0.0f;
    double proposal_log_probability = 0.0;
};

/*
 * One dynamically forced local selection frontier. It is not a persistent
 * syntax tree: the backend may own K live child KV states only until release()
 * is called. The product copies just BoundValue identities into its result.
 */
struct ProductNode {
    NodeId node_id = 0;
    std::size_t position = 0;
    std::vector<BoundContinuation> alternatives;
};

struct BoundPath {
    std::vector<BoundValue> positions;

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

/*
 * One projective/affine covector over a node's complete common support.
 * `boundary_count` is the mass retained by the associative scan; coordinates
 * are the corresponding arithmetic means in logit space (geometric means
 * after softmax).  The full-vocabulary partition fixes the projective gauge.
 */
struct PositionCovector {
    std::size_t position = 0;
    ObservationFrame frame;
    std::size_t boundary_count = 0;
    double vocabulary_log_partition =
        std::numeric_limits<double>::quiet_NaN();
    std::vector<FramedCoordinate> coordinates;
};

/* R in J_R X: a zip of covectors, never a fold or path score. */
struct ObservationTuple {
    std::vector<PositionCovector> positions;
};

/*
 * One restricted root-observer argument p(x : b(x)). `candidate` names x and
 * selected_suffix fixes b(x). The backend teacher-forces that completed
 * counterfactual and scans every native vocabulary covector at its causal
 * boundaries, including the history boundary before x and the boundary after
 * the final suffix token. It returns their affine mean. `common_support`
 * identifies the coordinates needed by the finite local selection; the
 * partition is nevertheless computed over the complete vocabulary.
 *
 * The scan carrier is (sum, mass), with componentwise sum and integer mass.
 * Thus recursive grouping cannot change the result. Covectors for different
 * completed counterfactuals remain different frames and their raw coordinates
 * are never compared directly.
 */
struct ObservationDemand {
    DemandId demand_id = 0;
    BoundValue candidate;
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

using BoundaryScanLaneId = std::uint64_t;

/*
 * Stable lowering identity for one completed counterfactual scan. The backend
 * starts with the history proposal, copies the history KV state to a lane,
 * teacher-forces candidate_then_suffix, and merges every resulting proposal
 * componentwise while retaining the exact boundary mass. There is one lane
 * per demand, including at the terminal selection position.
 */
struct BoundaryScanLane {
    BoundaryScanLaneId lane_id = 0;
    DemandId demand_id = 0;
    StructuredOutcomeRef history_before_candidate;
    std::vector<Token> candidate_then_suffix;
};

struct BoundaryScanSchedule {
    SelectionId selection_id = 0;
    std::size_t position = 0;
    std::vector<BoundaryScanLane> lanes;
};

/* Expands demands without evaluating or rating them. */
BoundaryScanSchedule make_boundary_scan_schedule(
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

/*
 * The model transition part of the term, separate from its one RootObserver.
 * demand() exposes one finite local support without running a candidate.
 * force() then materializes the K child histories in one backend batch.  The
 * split lets the product publish every continuation demand before any of those
 * continuations begins execution. force() is never called at the terminal
 * position.
 *
 * release() is called exactly once for every successfully returned node, once
 * all recursively selected suffixes have been copied into owning BoundPaths.
 * It must release every child state owned by that node, retain the binding
 * identity/token/rank metadata until the call returns, and must not throw.
 * Thus the runtime retains O(K * horizon) live frontier states, never K^N.
 */
class SelectionTerm {
public:
    virtual ~SelectionTerm() = default;

    virtual ProductNode demand(
        const StructuredOutcomeRef & history,
        std::size_t position
    ) = 0;

    virtual void force(
        const StructuredOutcomeRef & history,
        ProductNode & node
    ) = 0;

    virtual void release(ProductNode & node) noexcept = 0;
};

struct ObservedAlternative {
    const BoundValue * binding = nullptr;
    const BoundPath * selected_suffix = nullptr;
    const BoundPath * complete_path = nullptr;
    const PositionCovector * current_covector = nullptr;
    const ObservationTuple * observation_tuple = nullptr;
    double scan_ev_log_probability = 0.0;
    bool attains = false;
};

class ProductEventSink {
public:
    virtual ~ProductEventSink() = default;

    virtual void continuation_demanded(
        SelectionId /* selection_id */,
        std::size_t /* ordinal */,
        const BoundContinuation & /* binding */
    ) { }

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
    std::uint64_t attaining_alternatives = 0;
    std::uint64_t ambiguous_selection_nodes = 0;
    std::uint64_t zero_attaining_selection_nodes = 0;
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
 * For this finite-search epsilon, ev evaluates each x in the recursively
 * scanned boundary covector of its completed counterfactual p(x):
 *
 *   scan(p(x)) = mean_j boundary_logits_j(p(x))
 *   ev(x,p(x)) = log_softmax(scan(p(x)))[x].
 *
 * The scan composes vocabulary coordinates before ev; it never folds selected
 * token probabilities into a scalar path score. Carrying its boundary mass
 * makes it associative, and full-vocabulary normalization fixes the additive
 * gauge before evaluations from distinct frames are ordered. The selected
 * current covector is prepended to b(a)'s position-indexed tuple.
 */
class ExactProduct {
public:
    ExactProduct(
        std::size_t horizon,
        SelectionTerm & selection_term,
        RootObserver & root_observer,
        ProductEventSink * events = nullptr
    );

    ProductResult run(const StructuredOutcomeRef & root_history);
    const ProductCounters & counters() const;

private:
    ProductResult evaluate(
        const StructuredOutcomeRef & history,
        std::size_t position
    );

    std::vector<PositionCovector> observe_restrictions(
        SelectionId selection_id,
        const StructuredOutcomeRef & history,
        const ProductNode & node,
        const std::vector<BoundPath> & selected_suffixes
    );

    std::size_t horizon_ = 0;
    SelectionTerm & selection_term_;
    RootObserver & root_observer_;
    ProductEventSink * events_ = nullptr;
    ProductCounters counters_;
    SelectionId next_selection_id_ = 1;
    DemandId next_demand_id_ = 1;
};

} // namespace escardo_product

#endif
