#include "escardo_product_runtime.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace escardo_product {
namespace {

[[noreturn]] void invalid(const std::string & message) {
    throw std::invalid_argument("Escardo product: " + message);
}

void validate_outcome(
    const StructuredOutcomeRef & outcome,
    const std::string & owner
) {
    if (outcome.kv_summary.handle == 0 ||
        outcome.kv_summary.position < 0) {
        invalid(owner + " has no KV summary");
    }
    if (outcome.final_hidden.handle == 0 ||
        outcome.final_hidden.data == nullptr ||
        outcome.final_hidden.width == 0) {
        invalid(owner + " has no final hidden state");
    }
    if (outcome.proposal.handle == 0 ||
        outcome.proposal.data == nullptr ||
        outcome.proposal.width == 0) {
        invalid(owner + " has no proposal covector");
    }
}

void validate_suffix(
    const BoundPath & suffix,
    std::size_t begin,
    std::size_t horizon
) {
    if (suffix.positions.size() != horizon - begin) {
        invalid("selected suffix has the wrong length");
    }
    for (std::size_t index = 0; index < suffix.positions.size(); ++index) {
        const BoundContinuation * binding = suffix.positions[index];
        if (binding == nullptr) invalid("selected suffix has a null binding");
        if (binding->position != begin + index) {
            invalid("selected suffix is not position indexed");
        }
        validate_outcome(
            binding->outcome,
            "binding " + std::to_string(binding->binding_id)
        );
    }
}

const FramedCoordinate & coordinate_for(
    const PositionCovector & covector,
    Token token
) {
    const FramedCoordinate * found = nullptr;
    for (const FramedCoordinate & coordinate : covector.coordinates) {
        if (coordinate.token != token) continue;
        if (found != nullptr) invalid("covector repeats a support token");
        found = &coordinate;
    }
    if (found == nullptr) invalid("covector omits a support token");
    return *found;
}

bool greater_same_frame(
    const FramedCoordinate & left,
    const FramedCoordinate & right
) {
    if (left.frame != right.frame) {
        invalid("attempted to compare coordinates from different frames");
    }
    return left.value > right.value;
}

Token argmax_token(
    const PositionCovector & covector,
    const std::vector<Token> & support
) {
    if (support.empty()) invalid("cannot attain over an empty support");
    const FramedCoordinate * best = &coordinate_for(covector, support.front());
    for (std::size_t index = 1; index < support.size(); ++index) {
        const FramedCoordinate & candidate =
            coordinate_for(covector, support[index]);
        if (greater_same_frame(candidate, *best)) best = &candidate;
    }
    return best->token;
}

} // namespace

std::vector<Token> BoundPath::tokens() const {
    std::vector<Token> result;
    result.reserve(positions.size());
    for (const BoundContinuation * binding : positions) {
        if (binding == nullptr) invalid("cannot decode a null path binding");
        result.push_back(binding->token);
    }
    return result;
}

bool ObservationFrame::operator==(const ObservationFrame & other) const {
    return observer_id == other.observer_id && frame_id == other.frame_id;
}

bool ObservationFrame::operator!=(const ObservationFrame & other) const {
    return !(*this == other);
}

CausalPosteriorSchedule make_causal_posterior_schedule(
    const ObservationBatch & batch
) {
    if (batch.selection_id == 0) invalid("schedule has no selection id");
    if (batch.demands.empty()) invalid("schedule has no observer demands");

    CausalPosteriorSchedule schedule;
    schedule.selection_id = batch.selection_id;
    schedule.position = batch.selecting_position;
    PosteriorLaneId next_lane_id = 1;
    for (const ObservationDemand & demand : batch.demands) {
        if (demand.demand_id == 0 || demand.candidate == nullptr) {
            invalid("schedule contains an unbound observer demand");
        }
        validate_outcome(
            demand.history_before_candidate,
            "posterior lane history"
        );
        const std::vector<Token> suffix = demand.selected_suffix.tokens();
        for (Token candidate : demand.common_support) {
            CausalPosteriorLane lane;
            lane.lane_id = next_lane_id++;
            lane.demand_id = demand.demand_id;
            lane.candidate_token = candidate;
            lane.history_before_candidate =
                demand.history_before_candidate;
            lane.candidate_then_suffix.reserve(1 + suffix.size());
            lane.candidate_then_suffix.push_back(candidate);
            lane.candidate_then_suffix.insert(
                lane.candidate_then_suffix.end(),
                suffix.begin(), suffix.end()
            );
            lane.proposal_only = suffix.empty();
            schedule.lanes.push_back(std::move(lane));
        }
    }
    return schedule;
}

ExactProduct::ExactProduct(
    std::size_t horizon,
    RootObserver & root_observer,
    ProductEventSink * events
) :
    horizon_(horizon),
    root_observer_(root_observer),
    events_(events) {
    if (horizon_ == 0) invalid("horizon must be positive");
    if (root_observer_.observer_id() == 0) {
        invalid("root observer id zero is reserved");
    }
}

ProductResult ExactProduct::run(const ProductNode & root) {
    if (root.position != 0) invalid("root node must be at position zero");
    counters_ = ProductCounters{};
    next_selection_id_ = 1;
    next_demand_id_ = 1;
    std::vector<const ProductNode *> active;
    active.reserve(horizon_);
    return evaluate(root, active);
}

const ProductCounters & ExactProduct::counters() const {
    return counters_;
}

std::vector<PositionCovector> ExactProduct::observe_restrictions(
    SelectionId selection_id,
    const ProductNode & node,
    const std::vector<BoundPath> & selected_suffixes
) {
    if (selected_suffixes.size() != node.alternatives.size()) {
        invalid("Select lost a constructor/continuation binding");
    }
    validate_outcome(node.history, "selection history");

    std::vector<Token> support;
    support.reserve(node.alternatives.size());
    for (const BoundContinuation & binding : node.alternatives) {
        support.push_back(binding.token);
    }

    ObservationBatch batch;
    batch.selection_id = selection_id;
    batch.selecting_position = node.position;
    batch.demands.reserve(node.alternatives.size());
    for (std::size_t index = 0; index < node.alternatives.size(); ++index) {
        validate_suffix(
            selected_suffixes[index], node.position + 1, horizon_
        );
        ObservationDemand demand;
        demand.demand_id = next_demand_id_++;
        demand.candidate = &node.alternatives[index];
        demand.history_before_candidate = node.history;
        demand.selected_suffix = selected_suffixes[index];
        demand.common_support = support;
        batch.demands.push_back(std::move(demand));
    }

    if (events_ != nullptr) events_->observation_batch_requested(batch);
    counters_.observer_batches++;
    counters_.observer_demands += batch.demands.size();
    ObservationBatchResult result = root_observer_.observe(batch);
    if (result.selection_id != selection_id) {
        invalid("root observer returned the wrong selection id");
    }
    if (result.observations.size() != batch.demands.size()) {
        invalid("root observer returned the wrong demand count");
    }

    std::unordered_map<DemandId, std::size_t> by_id;
    by_id.reserve(result.observations.size());
    for (std::size_t index = 0; index < result.observations.size(); ++index) {
        const DemandId id = result.observations[index].demand_id;
        if (id == 0 || !by_id.emplace(id, index).second) {
            invalid("root observer returned a duplicate demand id");
        }
    }

    std::vector<PositionCovector> ordered;
    ordered.reserve(batch.demands.size());
    for (const ObservationDemand & demand : batch.demands) {
        const auto member = by_id.find(demand.demand_id);
        if (member == by_id.end()) {
            invalid("root observer omitted a restricted argument");
        }
        PositionCovector covector = std::move(
            result.observations[member->second].covector
        );
        if (covector.position != node.position) {
            invalid("root observer returned a covector at the wrong position");
        }
        if (covector.frame.observer_id != root_observer_.observer_id() ||
            covector.frame.frame_id == 0) {
            invalid("root observer returned an invalid frame");
        }
        if (covector.coordinates.size() != support.size()) {
            invalid("root observer did not return the complete common support");
        }
        for (Token token : support) {
            const FramedCoordinate & coordinate =
                coordinate_for(covector, token);
            if (coordinate.frame != covector.frame) {
                invalid("covector contains coordinates from another frame");
            }
            if (!std::isfinite(coordinate.value)) {
                invalid("root observer returned a non-finite coordinate");
            }
        }
        ordered.push_back(std::move(covector));
    }
    return ordered;
}

ProductResult ExactProduct::evaluate(
    const ProductNode & node,
    std::vector<const ProductNode *> & active
) {
    if (node.node_id == 0) invalid("node id zero is reserved");
    if (node.position >= horizon_) invalid("node lies beyond the horizon");
    if (node.alternatives.empty()) invalid("selection node has empty support");
    validate_outcome(node.history, "node history");
    if (std::find(active.begin(), active.end(), &node) != active.end()) {
        invalid("selection-function tree contains a cycle");
    }
    active.push_back(&node);
    counters_.selection_nodes++;
    const SelectionId selection_id = next_selection_id_++;

    std::unordered_set<BindingId> local_ids;
    std::unordered_set<Token> local_tokens;
    local_ids.reserve(node.alternatives.size());
    local_tokens.reserve(node.alternatives.size());
    std::vector<BoundPath> suffixes;
    std::vector<ObservationTuple> suffix_observations;
    suffixes.reserve(node.alternatives.size());
    suffix_observations.reserve(node.alternatives.size());
    for (std::size_t index = 0; index < node.alternatives.size(); ++index) {
        const BoundContinuation & binding = node.alternatives[index];
        if (binding.binding_id == 0 ||
            !local_ids.insert(binding.binding_id).second) {
            invalid("selection node has an invalid or duplicate binding id");
        }
        if (!local_tokens.insert(binding.token).second) {
            invalid("selection support repeats a token coordinate");
        }
        if (binding.local_rank != static_cast<std::int32_t>(index + 1)) {
            invalid("selection support is not in local-rank order");
        }
        if (binding.position != node.position) {
            invalid("candidate binding is attached to the wrong position");
        }
        validate_outcome(
            binding.outcome,
            "binding " + std::to_string(binding.binding_id)
        );

        if (node.position + 1 == horizon_) {
            if (binding.continuation != nullptr) {
                invalid("last-position binding has a continuation");
            }
            suffixes.emplace_back();
            suffix_observations.emplace_back();
        } else {
            if (binding.continuation == nullptr) {
                invalid("nonterminal binding has no continuation");
            }
            if (binding.continuation->position != node.position + 1) {
                invalid("continuation does not advance exactly one position");
            }
            ProductResult suffix = evaluate(*binding.continuation, active);
            validate_suffix(suffix.path, node.position + 1, horizon_);
            if (suffix.observation.positions.size() !=
                    horizon_ - node.position - 1) {
                invalid("recursive observer tuple was folded or truncated");
            }
            suffixes.push_back(std::move(suffix.path));
            suffix_observations.push_back(std::move(suffix.observation));
        }
    }

    std::vector<PositionCovector> current = observe_restrictions(
        selection_id, node, suffixes
    );
    std::vector<BoundPath> complete_paths;
    std::vector<ObservationTuple> complete_observations;
    std::vector<ObservedAlternative> alternatives;
    complete_paths.reserve(node.alternatives.size());
    complete_observations.reserve(node.alternatives.size());
    alternatives.reserve(node.alternatives.size());

    std::vector<Token> support;
    support.reserve(node.alternatives.size());
    for (const BoundContinuation & binding : node.alternatives) {
        support.push_back(binding.token);
    }
    std::size_t first_attaining = node.alternatives.size();
    for (std::size_t index = 0; index < node.alternatives.size(); ++index) {
        BoundPath path;
        path.positions.reserve(1 + suffixes[index].positions.size());
        path.positions.push_back(&node.alternatives[index]);
        path.positions.insert(
            path.positions.end(),
            suffixes[index].positions.begin(), suffixes[index].positions.end()
        );
        complete_paths.push_back(std::move(path));

        ObservationTuple tuple;
        tuple.positions.reserve(
            1 + suffix_observations[index].positions.size()
        );
        tuple.positions.push_back(std::move(current[index]));
        tuple.positions.insert(
            tuple.positions.end(),
            std::make_move_iterator(
                suffix_observations[index].positions.begin()
            ),
            std::make_move_iterator(
                suffix_observations[index].positions.end()
            )
        );
        complete_observations.push_back(std::move(tuple));

        const Token attained = argmax_token(
            complete_observations.back().positions.front(), support
        );
        const bool satisfies = attained == node.alternatives[index].token;
        if (satisfies && first_attaining == node.alternatives.size()) {
            first_attaining = index;
        }

        ObservedAlternative alternative;
        alternative.binding = &node.alternatives[index];
        alternative.selected_suffix = &suffixes[index];
        alternative.complete_path = &complete_paths[index];
        alternative.current_covector =
            &complete_observations[index].positions.front();
        alternative.observation_tuple = &complete_observations[index];
        alternative.attains = satisfies;
        alternatives.push_back(alternative);
        if (events_ != nullptr) {
            events_->candidate_observed(selection_id, alternatives.back());
        }
    }

    const std::size_t selected_index =
        first_attaining != node.alternatives.size()
            ? first_attaining
            : node.alternatives.size() - 1;
    if (events_ != nullptr) {
        events_->continuation_selected(
            selection_id, alternatives[selected_index]
        );
    }

    ProductResult selected;
    selected.path = std::move(complete_paths[selected_index]);
    selected.observation = std::move(complete_observations[selected_index]);
    active.pop_back();
    return selected;
}

} // namespace escardo_product
