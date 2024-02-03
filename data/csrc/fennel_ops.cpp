#include <torch/extension.h>
#include "fennel.hpp"

TORCH_LIBRARY(Fennel, m)
{
    m.def(
        "partition(Tensor rowptr, Tensor col, int k, Tensor? order, "
        "float gamma, float alpha, float slack, Tensor? init) -> Tensor",
        fennel::partition);
    m.def(
        "partition_opt(Tensor rowptr, Tensor col, int k, Tensor? order, "
        "float gamma, float alpha, float slack, Tensor? init, float? scan_thres) -> Tensor",
        fennel::partition_opt);

    /*
    m.def(
        "partition_parallel(Tensor rowptr, Tensor col, int k, Tensor? order, "
        "float gamma, float alpha, float slack, Tensor? init, float? scan_thres) -> Tensor",
        fennel::partition_par);
    m.def(
        "partition_strata(Tensor rowptr, Tensor col, int k, Tensor? labels, Tensor? order,"
        "float gamma, Tensor alphas, float slack, float l_slack, Tensor? init) -> Tensor",
        fennel::partition_stratified);
    m.def(
        "partition_strata_par(Tensor rowptr, Tensor col, int k, Tensor? labels, Tensor? order,"
        "float gamma, Tensor alphas, float slack, Tensor? init, float? scan_thres) -> Tensor",
        fennel::partition_stratified_par);
    */

    m.def(
        "partition_strata_opt(Tensor rowptr, Tensor col, int k, Tensor? labels, Tensor? order,"
        "float gamma, Tensor alphas, float slack, float l_slack, Tensor? init, float? scan_thres) -> Tensor",
        fennel::partition_stratified_opt);

    m.def(
        "partition_weighted(Tensor rowptr, Tensor col, Tensor? weights, int k, Tensor? order,"
        "float gamma, float alpha, float slack, Tensor? init, float? scan_thres) -> Tensor",
        fennel::partition_weighted
    );

    m.def(
        "partition_combined("
        "Tensor rowptr, Tensor col, Tensor? weights, int k, Tensor? order,"
        "float gamma, Tensor alphas, float slack, float slack2,"
        "Tensor strata_labels, Tensor balance_labels,Tensor? init, float? scan_thres"
        ") -> Tensor",
        fennel::partition_stratified_balanced_weighted
    );
}