abstract type DistributionsPreMetric <: Distances.PreMetric end
abstract type DistributionsSemiMetric <: Distances.SemiMetric end
abstract type DistributionsMetric <: Distances.Metric end

const DistributionsDistance = Union{
    DistributionsPreMetric,DistributionsSemiMetric,DistributionsMetric
}
