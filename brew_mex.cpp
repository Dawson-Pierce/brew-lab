// brew_mex.cpp — MEX gateway bridging MATLAB to the brew C++ library.
// Usage:  handle = brew_mex('command', args...)
//
// Commands:
//   Dynamics:     'create_dynamics', type_str -> handle
//   Filters:      'create_filter', type_str, dyn_handle, Q, H, R [, L] -> handle
//   Mixtures:     'create_mixture', dist_type, ...) -> handle
//   DBSCAN:       'create_dbscan', epsilon, min_pts -> handle
//   RFS filters:  'create_rfs', rfs_type, dist_type, filter_h, birth_h, params_struct -> handle
//   RFS ops:      'rfs_predict', handle, dt
//                 'rfs_correct', handle, measurements
//                 'rfs_cleanup_and_extract', handle -> mixture_struct
//                 'rfs_extract', handle -> mixture_struct
//   Getters:      'rfs_get_cardinality', handle -> vector
//                 'rfs_get_track_histories', handle -> struct
//   Cleanup:      'destroy', handle
//                 'destroy_all'

#include "mex.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <functional>

#include <Eigen/Dense>

// brew headers
#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/models/trajectory_ggiw.hpp"
#include "brew/models/mixture.hpp"
#include "brew/models/bernoulli.hpp"

#include "brew/dynamics/integrator_1d.hpp"
#include "brew/dynamics/integrator_2d.hpp"
#include "brew/dynamics/integrator_3d.hpp"
#include "brew/dynamics/double_integrator_1d.hpp"
#include "brew/dynamics/double_integrator_2d.hpp"
#include "brew/dynamics/double_integrator_3d.hpp"
#include "brew/dynamics/constant_turn_2d.hpp"

#include "brew/filters/ekf.hpp"
#include "brew/filters/ggiw_ekf.hpp"
#include "brew/filters/trajectory_gaussian_ekf.hpp"
#include "brew/filters/trajectory_ggiw_ekf.hpp"

#include "brew/multi_target/phd.hpp"
#include "brew/multi_target/cphd.hpp"
#include "brew/multi_target/mb.hpp"
#include "brew/multi_target/mbm.hpp"
#include "brew/multi_target/lmb.hpp"
#include "brew/multi_target/glmb.hpp"
#include "brew/multi_target/jglmb.hpp"
#include "brew/multi_target/pmbm.hpp"

#include "brew/clustering/dbscan.hpp"

using namespace brew;

// ============================================================
// Object Store
// ============================================================

struct ObjEntry {
    std::shared_ptr<void> ptr;
    std::string type;      // e.g. "Integrator1D", "EKF", "PHD", etc.
    std::string dist_type; // for filters/RFS: "Gaussian", "TrajectoryGaussian", etc.
    std::string rfs_type;  // for RFS filters: "PHD", "CPHD", etc.
    int timestep = 0;      // internal counter for RFS predict
};

static std::map<uint64_t, ObjEntry> g_store;
static uint64_t g_next_id = 1;

static uint64_t store_obj(std::shared_ptr<void> ptr, const std::string& type,
                           const std::string& dist_type = "",
                           const std::string& rfs_type = "") {
    uint64_t id = g_next_id++;
    g_store[id] = {ptr, type, dist_type, rfs_type, 0};
    return id;
}

template<typename T>
static std::shared_ptr<T> get_obj(uint64_t id) {
    auto it = g_store.find(id);
    if (it == g_store.end())
        mexErrMsgIdAndTxt("brew:badHandle", "Invalid object handle %llu.", id);
    return std::static_pointer_cast<T>(it->second.ptr);
}

static ObjEntry& get_entry(uint64_t id) {
    auto it = g_store.find(id);
    if (it == g_store.end())
        mexErrMsgIdAndTxt("brew:badHandle", "Invalid object handle %llu.", id);
    return it->second;
}

static void cleanup_all() { g_store.clear(); }

// ============================================================
// MATLAB <-> C++ Conversion Helpers
// ============================================================

static std::string get_string(const mxArray* a) {
    // Handle char arrays directly
    if (mxIsChar(a)) {
        char* buf = mxArrayToUTF8String(a);
        std::string s(buf);
        mxFree(buf);
        return s;
    }
    // Handle MATLAB string scalars (R2017b+) by calling char()
    mxArray* tmp = nullptr;
    mxArray* input = const_cast<mxArray*>(a);
    if (mexCallMATLAB(1, &tmp, 1, &input, "char") == 0 && tmp && mxIsChar(tmp)) {
        char* buf = mxArrayToUTF8String(tmp);
        std::string s(buf);
        mxFree(buf);
        mxDestroyArray(tmp);
        return s;
    }
    if (tmp) mxDestroyArray(tmp);
    mexErrMsgIdAndTxt("brew:badArg", "Expected a string argument.");
    return "";
}

static uint64_t get_handle(const mxArray* a) {
    if (mxIsUint64(a)) return *static_cast<uint64_t*>(mxGetData(a));
    return static_cast<uint64_t>(mxGetScalar(a));
}

static mxArray* make_handle(uint64_t h) {
    mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *static_cast<uint64_t*>(mxGetData(out)) = h;
    return out;
}

static Eigen::VectorXd to_eigen_vec(const mxArray* a) {
    size_t n = mxGetNumberOfElements(a);
    double* p = mxGetDoubles(a);
    return Eigen::Map<Eigen::VectorXd>(p, static_cast<Eigen::Index>(n));
}

static Eigen::MatrixXd to_eigen_mat(const mxArray* a) {
    size_t m = mxGetM(a);
    size_t n = mxGetN(a);
    double* p = mxGetDoubles(a);
    return Eigen::Map<Eigen::MatrixXd>(p, static_cast<Eigen::Index>(m),
                                        static_cast<Eigen::Index>(n));
}

static mxArray* from_eigen_vec(const Eigen::VectorXd& v) {
    mxArray* a = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
    std::memcpy(mxGetDoubles(a), v.data(), v.size() * sizeof(double));
    return a;
}

static mxArray* from_eigen_mat(const Eigen::MatrixXd& M) {
    mxArray* a = mxCreateDoubleMatrix(M.rows(), M.cols(), mxREAL);
    std::memcpy(mxGetDoubles(a), M.data(), M.size() * sizeof(double));
    return a;
}

static double get_field_double(const mxArray* s, const char* name, double def) {
    mxArray* f = mxGetField(s, 0, name);
    return f ? mxGetScalar(f) : def;
}

static int get_field_int(const mxArray* s, const char* name, int def) {
    return static_cast<int>(get_field_double(s, name, static_cast<double>(def)));
}

// ============================================================
// Dynamics Creation
// ============================================================

static uint64_t cmd_create_dynamics(const std::string& type) {
    std::shared_ptr<dynamics::DynamicsBase> dyn;
    if      (type == "Integrator1D")        dyn = std::make_shared<dynamics::Integrator1D>();
    else if (type == "Integrator2D")        dyn = std::make_shared<dynamics::Integrator2D>();
    else if (type == "Integrator3D")        dyn = std::make_shared<dynamics::Integrator3D>();
    else if (type == "DoubleIntegrator1D")  dyn = std::make_shared<dynamics::DoubleIntegrator1D>();
    else if (type == "DoubleIntegrator2D")  dyn = std::make_shared<dynamics::DoubleIntegrator2D>();
    else if (type == "DoubleIntegrator3D")  dyn = std::make_shared<dynamics::DoubleIntegrator3D>();
    else if (type == "ConstantTurn2D")      dyn = std::make_shared<dynamics::ConstantTurn2D>();
    else mexErrMsgIdAndTxt("brew:badDyn", "Unknown dynamics type: %s", type.c_str());
    return store_obj(dyn, type);
}

// ============================================================
// DBSCAN Creation
// ============================================================

static uint64_t cmd_create_dbscan(double epsilon, int min_pts) {
    auto obj = std::make_shared<clustering::DBSCAN>(epsilon, min_pts);
    return store_obj(obj, "DBSCAN");
}

// ============================================================
// Filter Creation
// ============================================================

static uint64_t cmd_create_filter(int nrhs, const mxArray* prhs[]) {
    // prhs: [1]=filter_type, [2]=dyn_handle, [3]=Q, [4]=H, [5]=R, [6..]=optional
    if (nrhs < 6)
        mexErrMsgIdAndTxt("brew:badArgs", "create_filter needs: type, dyn_handle, Q, H, R");

    std::string ftype = get_string(prhs[1]);
    uint64_t dyn_h = get_handle(prhs[2]);
    Eigen::MatrixXd Q = to_eigen_mat(prhs[3]);
    Eigen::MatrixXd H = to_eigen_mat(prhs[4]);
    Eigen::MatrixXd R = to_eigen_mat(prhs[5]);

    auto dyn = get_obj<dynamics::DynamicsBase>(dyn_h);

    if (ftype == "EKF") {
        auto f = std::make_shared<filters::EKF>();
        f->set_dynamics(dyn);
        f->set_process_noise(Q);
        f->set_measurement_jacobian(H);
        f->set_measurement_noise(R);
        return store_obj(f, "EKF", "Gaussian");
    }
    else if (ftype == "GGIWEKF") {
        auto f = std::make_shared<filters::GGIWEKF>();
        f->set_dynamics(dyn);
        f->set_process_noise(Q);
        f->set_measurement_jacobian(H);
        f->set_measurement_noise(R);
        if (nrhs > 6) f->set_temporal_decay(mxGetScalar(prhs[6]));
        if (nrhs > 7) f->set_forgetting_factor(mxGetScalar(prhs[7]));
        if (nrhs > 8) f->set_scaling_parameter(mxGetScalar(prhs[8]));
        return store_obj(f, "GGIWEKF", "GGIW");
    }
    else if (ftype == "TrajectoryGaussianEKF") {
        auto f = std::make_shared<filters::TrajectoryGaussianEKF>();
        f->set_dynamics(dyn);
        f->set_process_noise(Q);
        f->set_measurement_jacobian(H);
        f->set_measurement_noise(R);
        if (nrhs > 6) f->set_window_size(static_cast<int>(mxGetScalar(prhs[6])));
        return store_obj(f, "TrajectoryGaussianEKF", "TrajectoryGaussian");
    }
    else if (ftype == "TrajectoryGGIWEKF") {
        auto f = std::make_shared<filters::TrajectoryGGIWEKF>();
        f->set_dynamics(dyn);
        f->set_process_noise(Q);
        f->set_measurement_jacobian(H);
        f->set_measurement_noise(R);
        if (nrhs > 6) f->set_window_size(static_cast<int>(mxGetScalar(prhs[6])));
        if (nrhs > 7) f->set_temporal_decay(mxGetScalar(prhs[7]));
        if (nrhs > 8) f->set_forgetting_factor(mxGetScalar(prhs[8]));
        if (nrhs > 9) f->set_scaling_parameter(mxGetScalar(prhs[9]));
        return store_obj(f, "TrajectoryGGIWEKF", "TrajectoryGGIW");
    }
    else {
        mexErrMsgIdAndTxt("brew:badFilter", "Unknown filter type: %s", ftype.c_str());
        return 0;
    }
}

// ============================================================
// Mixture Creation
// ============================================================

static uint64_t create_gaussian_mixture(int nrhs, const mxArray* prhs[]) {
    // prhs: [1]='Gaussian', [2]=means_cell, [3]=covs_cell, [4]=weights_vec
    if (nrhs < 5) mexErrMsgIdAndTxt("brew:badArgs", "Gaussian mixture: means, covs, weights");
    const mxArray* means_cell = prhs[2];
    const mxArray* covs_cell  = prhs[3];
    const mxArray* weights    = prhs[4];
    size_t n = mxGetNumberOfElements(means_cell);
    double* w = mxGetDoubles(weights);

    auto mix = std::make_shared<models::Mixture<models::Gaussian>>();
    for (size_t i = 0; i < n; ++i) {
        Eigen::VectorXd mean = to_eigen_vec(mxGetCell(means_cell, i));
        Eigen::MatrixXd cov  = to_eigen_mat(mxGetCell(covs_cell, i));
        mix->add_component(std::make_unique<models::Gaussian>(mean, cov), w[i]);
    }
    return store_obj(mix, "Mixture", "Gaussian");
}

static uint64_t create_ggiw_mixture(int nrhs, const mxArray* prhs[]) {
    // [1]='GGIW', [2]=means, [3]=covs, [4]=alphas, [5]=betas, [6]=vs, [7]=Vs, [8]=weights
    if (nrhs < 9) mexErrMsgIdAndTxt("brew:badArgs",
        "GGIW mixture: means, covs, alphas, betas, vs, Vs, weights");
    const mxArray* means_cell  = prhs[2];
    const mxArray* covs_cell   = prhs[3];
    const mxArray* alphas      = prhs[4];
    const mxArray* betas       = prhs[5];
    const mxArray* vs          = prhs[6];
    const mxArray* Vs_cell     = prhs[7];
    const mxArray* weights     = prhs[8];
    size_t n = mxGetNumberOfElements(means_cell);
    double* w  = mxGetDoubles(weights);
    double* pa = mxGetDoubles(alphas);
    double* pb = mxGetDoubles(betas);
    double* pv = mxGetDoubles(vs);

    auto mix = std::make_shared<models::Mixture<models::GGIW>>();
    for (size_t i = 0; i < n; ++i) {
        Eigen::VectorXd mean = to_eigen_vec(mxGetCell(means_cell, i));
        Eigen::MatrixXd cov  = to_eigen_mat(mxGetCell(covs_cell, i));
        Eigen::MatrixXd V    = to_eigen_mat(mxGetCell(Vs_cell, i));
        mix->add_component(
            std::make_unique<models::GGIW>(mean, cov, pa[i], pb[i], pv[i], V), w[i]);
    }
    return store_obj(mix, "Mixture", "GGIW");
}

static uint64_t create_traj_gaussian_mixture(int nrhs, const mxArray* prhs[]) {
    // [1]='TrajectoryGaussian', [2]=idx_cell, [3]=means_cell, [4]=covs_cell, [5]=weights
    if (nrhs < 6) mexErrMsgIdAndTxt("brew:badArgs",
        "TrajectoryGaussian mixture: idxs, means, covs, weights");
    const mxArray* idx_cell   = prhs[2];
    const mxArray* means_cell = prhs[3];
    const mxArray* covs_cell  = prhs[4];
    const mxArray* weights    = prhs[5];
    size_t n = mxGetNumberOfElements(means_cell);
    double* w = mxGetDoubles(weights);

    auto mix = std::make_shared<models::Mixture<models::TrajectoryGaussian>>();
    for (size_t i = 0; i < n; ++i) {
        int idx = static_cast<int>(mxGetScalar(mxGetCell(idx_cell, i)));
        Eigen::VectorXd mean = to_eigen_vec(mxGetCell(means_cell, i));
        Eigen::MatrixXd cov  = to_eigen_mat(mxGetCell(covs_cell, i));
        int state_dim = static_cast<int>(mean.size()); // initial: 1 step, so state_dim = mean.size()
        mix->add_component(
            std::make_unique<models::TrajectoryGaussian>(idx, state_dim, mean, cov), w[i]);
    }
    return store_obj(mix, "Mixture", "TrajectoryGaussian");
}

static uint64_t create_traj_ggiw_mixture(int nrhs, const mxArray* prhs[]) {
    // [1]='TrajectoryGGIW', [2]=idxs, [3]=means, [4]=covs, [5]=alphas, [6]=betas,
    // [7]=vs, [8]=Vs, [9]=weights
    if (nrhs < 10) mexErrMsgIdAndTxt("brew:badArgs",
        "TrajectoryGGIW mixture: idxs, means, covs, alphas, betas, vs, Vs, weights");
    const mxArray* idx_cell    = prhs[2];
    const mxArray* means_cell  = prhs[3];
    const mxArray* covs_cell   = prhs[4];
    const mxArray* alphas      = prhs[5];
    const mxArray* betas       = prhs[6];
    const mxArray* vs          = prhs[7];
    const mxArray* Vs_cell     = prhs[8];
    const mxArray* weights     = prhs[9];
    size_t n = mxGetNumberOfElements(means_cell);
    double* w  = mxGetDoubles(weights);
    double* pa = mxGetDoubles(alphas);
    double* pb = mxGetDoubles(betas);
    double* pv = mxGetDoubles(vs);

    auto mix = std::make_shared<models::Mixture<models::TrajectoryGGIW>>();
    for (size_t i = 0; i < n; ++i) {
        int idx = static_cast<int>(mxGetScalar(mxGetCell(idx_cell, i)));
        Eigen::VectorXd mean = to_eigen_vec(mxGetCell(means_cell, i));
        Eigen::MatrixXd cov  = to_eigen_mat(mxGetCell(covs_cell, i));
        Eigen::MatrixXd V    = to_eigen_mat(mxGetCell(Vs_cell, i));
        int state_dim = static_cast<int>(mean.size());
        mix->add_component(
            std::make_unique<models::TrajectoryGGIW>(
                idx, state_dim, mean, cov, pa[i], pb[i], pv[i], V), w[i]);
    }
    return store_obj(mix, "Mixture", "TrajectoryGGIW");
}

static uint64_t cmd_create_mixture(int nrhs, const mxArray* prhs[]) {
    std::string dt = get_string(prhs[1]);
    if      (dt == "Gaussian")            return create_gaussian_mixture(nrhs, prhs);
    else if (dt == "GGIW")                return create_ggiw_mixture(nrhs, prhs);
    else if (dt == "TrajectoryGaussian")   return create_traj_gaussian_mixture(nrhs, prhs);
    else if (dt == "TrajectoryGGIW")       return create_traj_ggiw_mixture(nrhs, prhs);
    else mexErrMsgIdAndTxt("brew:badDist", "Unknown dist type: %s", dt.c_str());
    return 0;
}

// ============================================================
// RFS Filter Creation (templated)
// ============================================================

// Configure base RFS parameters from a MATLAB struct
static void configure_rfs_base(multi_target::RFSBase& rfs, const mxArray* p) {
    rfs.set_prob_detection(get_field_double(p, "prob_detection", 0.9));
    rfs.set_prob_survive(get_field_double(p, "prob_survive", 0.99));
    rfs.set_clutter_rate(get_field_double(p, "clutter_rate", 0.0));
    rfs.set_clutter_density(get_field_double(p, "clutter_density", 0.0));
}

// ----- PHD -----
template<typename T>
static uint64_t create_phd_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {
    auto phd = std::make_shared<multi_target::PHD<T>>();
    configure_rfs_base(*phd, p);

    // Initialize intensity to empty mixture (required — predict guards on nullptr)
    phd->set_intensity(std::make_unique<models::Mixture<T>>());

    auto filt = get_obj<filters::Filter<T>>(filter_h);
    phd->set_filter(filt->clone());

    auto birth = get_obj<models::Mixture<T>>(birth_h);
    phd->set_birth_model(birth->clone());

    phd->set_prune_threshold(get_field_double(p, "prune_threshold", 1e-4));
    phd->set_merge_threshold(get_field_double(p, "merge_threshold", 4.0));
    phd->set_max_components(get_field_int(p, "max_components", 100));
    phd->set_extract_threshold(get_field_double(p, "extract_threshold", 0.5));
    phd->set_gate_threshold(get_field_double(p, "gate_threshold", 9.0));

    return store_obj(phd, "PHD", get_entry(filter_h).dist_type, "PHD");
}

// ----- CPHD -----
template<typename T>
static uint64_t create_cphd_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {
    auto rfs = std::make_shared<multi_target::CPHD<T>>();
    configure_rfs_base(*rfs, p);

    // Initialize intensity to empty mixture (required — predict guards on nullptr)
    rfs->set_intensity(std::make_unique<models::Mixture<T>>());

    auto filt = get_obj<filters::Filter<T>>(filter_h);
    rfs->set_filter(filt->clone());
    auto birth = get_obj<models::Mixture<T>>(birth_h);
    rfs->set_birth_model(birth->clone());

    rfs->set_prune_threshold(get_field_double(p, "prune_threshold", 1e-4));
    rfs->set_merge_threshold(get_field_double(p, "merge_threshold", 4.0));
    rfs->set_max_components(get_field_int(p, "max_components", 100));
    rfs->set_extract_threshold(get_field_double(p, "extract_threshold", 0.5));
    rfs->set_gate_threshold(get_field_double(p, "gate_threshold", 9.0));
    rfs->set_max_cardinality(get_field_int(p, "max_cardinality", 100));

    double poisson_card = get_field_double(p, "poisson_cardinality", -1);
    if (poisson_card > 0) rfs->set_poisson_cardinality(poisson_card);
    double poisson_birth = get_field_double(p, "poisson_birth_cardinality", -1);
    if (poisson_birth > 0) rfs->set_poisson_birth_cardinality(poisson_birth);

    return store_obj(rfs, "CPHD", get_entry(filter_h).dist_type, "CPHD");
}

// ----- MB -----
template<typename T>
static uint64_t create_mb_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {
    auto rfs = std::make_shared<multi_target::MB<T>>();
    configure_rfs_base(*rfs, p);

    auto filt = get_obj<filters::Filter<T>>(filter_h);
    rfs->set_filter(filt->clone());
    auto birth = get_obj<models::Mixture<T>>(birth_h);
    rfs->set_birth_model(birth->clone());

    rfs->set_prune_threshold_bernoulli(get_field_double(p, "prune_threshold_bernoulli", 1e-3));
    rfs->set_extract_threshold(get_field_double(p, "extract_threshold", 0.5));
    rfs->set_gate_threshold(get_field_double(p, "gate_threshold", 9.0));

    return store_obj(rfs, "MB", get_entry(filter_h).dist_type, "MB");
}

// ----- MBM -----
template<typename T>
static uint64_t create_mbm_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {
    auto rfs = std::make_shared<multi_target::MBM<T>>();
    configure_rfs_base(*rfs, p);

    auto filt = get_obj<filters::Filter<T>>(filter_h);
    rfs->set_filter(filt->clone());
    auto birth = get_obj<models::Mixture<T>>(birth_h);
    rfs->set_birth_model(birth->clone());

    rfs->set_prune_threshold_hypothesis(get_field_double(p, "prune_threshold_hypothesis", 1e-3));
    rfs->set_prune_threshold_bernoulli(get_field_double(p, "prune_threshold_bernoulli", 1e-3));
    rfs->set_max_hypotheses(get_field_int(p, "max_hypotheses", 100));
    rfs->set_extract_threshold(get_field_double(p, "extract_threshold", 0.5));
    rfs->set_gate_threshold(get_field_double(p, "gate_threshold", 9.0));
    rfs->set_k_best(get_field_int(p, "k_best", 5));

    return store_obj(rfs, "MBM", get_entry(filter_h).dist_type, "MBM");
}

// ----- LMB -----
template<typename T>
static uint64_t create_lmb_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {
    auto rfs = std::make_shared<multi_target::LMB<T>>();
    configure_rfs_base(*rfs, p);

    auto filt = get_obj<filters::Filter<T>>(filter_h);
    rfs->set_filter(filt->clone());
    auto birth = get_obj<models::Mixture<T>>(birth_h);
    rfs->set_birth_model(birth->clone());

    rfs->set_prune_threshold_bernoulli(get_field_double(p, "prune_threshold_bernoulli", 1e-3));
    rfs->set_extract_threshold(get_field_double(p, "extract_threshold", 0.5));
    rfs->set_gate_threshold(get_field_double(p, "gate_threshold", 9.0));
    rfs->set_k_best(get_field_int(p, "k_best", 5));

    return store_obj(rfs, "LMB", get_entry(filter_h).dist_type, "LMB");
}

// ----- GLMB -----
template<typename T>
static uint64_t create_glmb_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {
    auto rfs = std::make_shared<multi_target::GLMB<T>>();
    configure_rfs_base(*rfs, p);

    auto filt = get_obj<filters::Filter<T>>(filter_h);
    rfs->set_filter(filt->clone());
    auto birth = get_obj<models::Mixture<T>>(birth_h);
    rfs->set_birth_model(birth->clone());

    rfs->set_prune_threshold_hypothesis(get_field_double(p, "prune_threshold_hypothesis", 1e-3));
    rfs->set_prune_threshold_bernoulli(get_field_double(p, "prune_threshold_bernoulli", 1e-3));
    rfs->set_max_hypotheses(get_field_int(p, "max_hypotheses", 100));
    rfs->set_extract_threshold(get_field_double(p, "extract_threshold", 0.5));
    rfs->set_gate_threshold(get_field_double(p, "gate_threshold", 9.0));
    rfs->set_k_best(get_field_int(p, "k_best", 5));

    return store_obj(rfs, "GLMB", get_entry(filter_h).dist_type, "GLMB");
}

// ----- JGLMB -----
template<typename T>
static uint64_t create_jglmb_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {
    auto rfs = std::make_shared<multi_target::JGLMB<T>>();
    configure_rfs_base(*rfs, p);

    auto filt = get_obj<filters::Filter<T>>(filter_h);
    rfs->set_filter(filt->clone());
    auto birth = get_obj<models::Mixture<T>>(birth_h);
    rfs->set_birth_model(birth->clone());

    rfs->set_prune_threshold_hypothesis(get_field_double(p, "prune_threshold_hypothesis", 1e-3));
    rfs->set_prune_threshold_bernoulli(get_field_double(p, "prune_threshold_bernoulli", 1e-3));
    rfs->set_max_hypotheses(get_field_int(p, "max_hypotheses", 100));
    rfs->set_extract_threshold(get_field_double(p, "extract_threshold", 0.5));
    rfs->set_gate_threshold(get_field_double(p, "gate_threshold", 9.0));
    rfs->set_k_best(get_field_int(p, "k_best", 5));

    return store_obj(rfs, "JGLMB", get_entry(filter_h).dist_type, "JGLMB");
}

// ----- PMBM -----
template<typename T>
static uint64_t create_pmbm_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {
    auto rfs = std::make_shared<multi_target::PMBM<T>>();
    configure_rfs_base(*rfs, p);

    // Initialize Poisson intensity to empty mixture (required — predict guards on nullptr)
    rfs->set_poisson_intensity(std::make_unique<models::Mixture<T>>());

    auto filt = get_obj<filters::Filter<T>>(filter_h);
    rfs->set_filter(filt->clone());
    auto birth = get_obj<models::Mixture<T>>(birth_h);
    rfs->set_birth_model(birth->clone());

    rfs->set_prune_poisson_threshold(get_field_double(p, "prune_poisson_threshold", 1e-4));
    rfs->set_merge_poisson_threshold(get_field_double(p, "merge_poisson_threshold", 4.0));
    rfs->set_max_poisson_components(get_field_int(p, "max_poisson_components", 100));
    rfs->set_prune_threshold_hypothesis(get_field_double(p, "prune_threshold_hypothesis", 1e-3));
    rfs->set_prune_threshold_bernoulli(get_field_double(p, "prune_threshold_bernoulli", 1e-3));
    rfs->set_recycle_threshold(get_field_double(p, "recycle_threshold", 0.1));
    rfs->set_max_hypotheses(get_field_int(p, "max_hypotheses", 100));
    rfs->set_extract_threshold(get_field_double(p, "extract_threshold", 0.5));
    rfs->set_gate_threshold(get_field_double(p, "gate_threshold", 9.0));
    rfs->set_k_best(get_field_int(p, "k_best", 5));

    return store_obj(rfs, "PMBM", get_entry(filter_h).dist_type, "PMBM");
}

// Dispatch: rfs_type -> dist_type -> create
template<typename T>
static uint64_t create_rfs_dispatch_dist(const std::string& rfs_type,
                                          uint64_t fh, uint64_t bh, const mxArray* p) {
    if      (rfs_type == "PHD")    return create_phd_impl<T>(fh, bh, p);
    else if (rfs_type == "CPHD")   return create_cphd_impl<T>(fh, bh, p);
    else if (rfs_type == "MB")     return create_mb_impl<T>(fh, bh, p);
    else if (rfs_type == "MBM")    return create_mbm_impl<T>(fh, bh, p);
    else if (rfs_type == "LMB")    return create_lmb_impl<T>(fh, bh, p);
    else if (rfs_type == "GLMB")   return create_glmb_impl<T>(fh, bh, p);
    else if (rfs_type == "JGLMB")  return create_jglmb_impl<T>(fh, bh, p);
    else if (rfs_type == "PMBM")   return create_pmbm_impl<T>(fh, bh, p);
    else mexErrMsgIdAndTxt("brew:badRFS", "Unknown RFS type: %s", rfs_type.c_str());
    return 0;
}

static uint64_t cmd_create_rfs(int nrhs, const mxArray* prhs[]) {
    // prhs: [1]=rfs_type, [2]=dist_type, [3]=filter_handle, [4]=birth_handle, [5]=params_struct
    if (nrhs < 6)
        mexErrMsgIdAndTxt("brew:badArgs",
            "create_rfs needs: rfs_type, dist_type, filter_h, birth_h, params");
    std::string rfs_type  = get_string(prhs[1]);
    std::string dist_type = get_string(prhs[2]);
    uint64_t fh = get_handle(prhs[3]);
    uint64_t bh = get_handle(prhs[4]);
    const mxArray* params = prhs[5];

    if      (dist_type == "Gaussian")
        return create_rfs_dispatch_dist<models::Gaussian>(rfs_type, fh, bh, params);
    else if (dist_type == "GGIW")
        return create_rfs_dispatch_dist<models::GGIW>(rfs_type, fh, bh, params);
    else if (dist_type == "TrajectoryGaussian")
        return create_rfs_dispatch_dist<models::TrajectoryGaussian>(rfs_type, fh, bh, params);
    else if (dist_type == "TrajectoryGGIW")
        return create_rfs_dispatch_dist<models::TrajectoryGGIW>(rfs_type, fh, bh, params);
    else
        mexErrMsgIdAndTxt("brew:badDist", "Unknown dist type: %s", dist_type.c_str());
    return 0;
}

// ============================================================
// RFS Operations (polymorphic via RFSBase)
// ============================================================

static void cmd_rfs_predict(uint64_t handle, double dt) {
    auto& entry = get_entry(handle);
    auto rfs = get_obj<multi_target::RFSBase>(handle);
    entry.timestep++;
    rfs->predict(entry.timestep, dt);
}

static void cmd_rfs_correct(uint64_t handle, const mxArray* meas_mx) {
    auto rfs = get_obj<multi_target::RFSBase>(handle);
    Eigen::MatrixXd meas;

    if (mxIsEmpty(meas_mx)) {
        meas.resize(1, 0);
    } else {
        size_t m = mxGetM(meas_mx);
        size_t n = mxGetN(meas_mx);
        // If it's a row vector, treat each element as a 1D measurement (1 x N)
        if (m == 1 && n > 1) {
            meas = to_eigen_mat(meas_mx); // 1 x N, each column is a measurement
        } else if (n == 1 && m > 1) {
            // Column vector: reshape to 1 x M (each element is a 1D measurement)
            Eigen::VectorXd v = to_eigen_vec(meas_mx);
            meas.resize(1, v.size());
            meas.row(0) = v.transpose();
        } else {
            meas = to_eigen_mat(meas_mx); // M x N, each column is an M-dim measurement
        }
    }
    rfs->correct(meas);
}

static void cmd_rfs_cleanup(uint64_t handle) {
    auto rfs = get_obj<multi_target::RFSBase>(handle);
    rfs->cleanup();
}

// ============================================================
// Mixture Extraction (template dispatch to convert to MATLAB)
// ============================================================

// --- Gaussian ---
static mxArray* mixture_to_matlab(const models::Mixture<models::Gaussian>& mix) {
    size_t n = mix.size();
    const char* fields[] = {"means", "covariances", "weights", "dist_type"};
    mxArray* s = mxCreateStructMatrix(1, 1, 4, fields);

    mxArray* mc = mxCreateCellMatrix(1, n);
    mxArray* cc = mxCreateCellMatrix(1, n);
    for (size_t i = 0; i < n; ++i) {
        mxSetCell(mc, i, from_eigen_vec(mix.component(i).mean()));
        mxSetCell(cc, i, from_eigen_mat(mix.component(i).covariance()));
    }
    mxSetField(s, 0, "means", mc);
    mxSetField(s, 0, "covariances", cc);
    mxSetField(s, 0, "weights", from_eigen_vec(mix.weights()));
    mxSetField(s, 0, "dist_type", mxCreateString("Gaussian"));
    return s;
}

// --- TrajectoryGaussian ---
static mxArray* mixture_to_matlab(
        const models::Mixture<models::TrajectoryGaussian>& mix) {
    size_t n = mix.size();
    const char* fields[] = {"means", "covariances", "weights", "dist_type",
                            "state_dim", "init_indices", "mean_histories"};
    mxArray* s = mxCreateStructMatrix(1, 1, 7, fields);

    mxArray* mc = mxCreateCellMatrix(1, n);
    mxArray* cc = mxCreateCellMatrix(1, n);
    mxArray* hc = mxCreateCellMatrix(1, n);
    mxArray* idxs = mxCreateDoubleMatrix(1, n, mxREAL);
    double* pidx = mxGetDoubles(idxs);
    int sd = 0;

    for (size_t i = 0; i < n; ++i) {
        const auto& comp = mix.component(i);
        mxSetCell(mc, i, from_eigen_vec(comp.mean()));
        mxSetCell(cc, i, from_eigen_mat(comp.covariance()));
        mxSetCell(hc, i, from_eigen_mat(comp.mean_history()));
        pidx[i] = static_cast<double>(comp.init_idx);
        sd = comp.state_dim;
    }
    mxSetField(s, 0, "means", mc);
    mxSetField(s, 0, "covariances", cc);
    mxSetField(s, 0, "weights", from_eigen_vec(mix.weights()));
    mxSetField(s, 0, "dist_type", mxCreateString("TrajectoryGaussian"));
    mxSetField(s, 0, "state_dim", mxCreateDoubleScalar(sd));
    mxSetField(s, 0, "init_indices", idxs);
    mxSetField(s, 0, "mean_histories", hc);
    return s;
}

// --- GGIW ---
static mxArray* mixture_to_matlab(const models::Mixture<models::GGIW>& mix) {
    size_t n = mix.size();
    const char* fields[] = {"means", "covariances", "weights", "dist_type",
                            "alphas", "betas", "vs", "Vs"};
    mxArray* s = mxCreateStructMatrix(1, 1, 8, fields);

    mxArray* mc = mxCreateCellMatrix(1, n);
    mxArray* cc = mxCreateCellMatrix(1, n);
    mxArray* Vc = mxCreateCellMatrix(1, n);
    mxArray* alp = mxCreateDoubleMatrix(1, n, mxREAL);
    mxArray* bet = mxCreateDoubleMatrix(1, n, mxREAL);
    mxArray* vv  = mxCreateDoubleMatrix(1, n, mxREAL);
    double* pa = mxGetDoubles(alp);
    double* pb = mxGetDoubles(bet);
    double* pv = mxGetDoubles(vv);

    for (size_t i = 0; i < n; ++i) {
        const auto& comp = mix.component(i);
        mxSetCell(mc, i, from_eigen_vec(comp.mean()));
        mxSetCell(cc, i, from_eigen_mat(comp.covariance()));
        mxSetCell(Vc, i, from_eigen_mat(comp.V()));
        pa[i] = comp.alpha();
        pb[i] = comp.beta();
        pv[i] = comp.v();
    }
    mxSetField(s, 0, "means", mc);
    mxSetField(s, 0, "covariances", cc);
    mxSetField(s, 0, "weights", from_eigen_vec(mix.weights()));
    mxSetField(s, 0, "dist_type", mxCreateString("GGIW"));
    mxSetField(s, 0, "alphas", alp);
    mxSetField(s, 0, "betas", bet);
    mxSetField(s, 0, "vs", vv);
    mxSetField(s, 0, "Vs", Vc);
    return s;
}

// --- TrajectoryGGIW ---
static mxArray* mixture_to_matlab(
        const models::Mixture<models::TrajectoryGGIW>& mix) {
    size_t n = mix.size();
    const char* fields[] = {"means", "covariances", "weights", "dist_type",
                            "state_dim", "init_indices", "mean_histories",
                            "alphas", "betas", "vs", "Vs"};
    mxArray* s = mxCreateStructMatrix(1, 1, 11, fields);

    mxArray* mc = mxCreateCellMatrix(1, n);
    mxArray* cc = mxCreateCellMatrix(1, n);
    mxArray* hc = mxCreateCellMatrix(1, n);
    mxArray* Vc = mxCreateCellMatrix(1, n);
    mxArray* idxs = mxCreateDoubleMatrix(1, n, mxREAL);
    mxArray* alp  = mxCreateDoubleMatrix(1, n, mxREAL);
    mxArray* bet  = mxCreateDoubleMatrix(1, n, mxREAL);
    mxArray* vv   = mxCreateDoubleMatrix(1, n, mxREAL);
    double* pidx = mxGetDoubles(idxs);
    double* pa   = mxGetDoubles(alp);
    double* pb   = mxGetDoubles(bet);
    double* pv   = mxGetDoubles(vv);
    int sd = 0;

    for (size_t i = 0; i < n; ++i) {
        const auto& comp = mix.component(i);
        mxSetCell(mc, i, from_eigen_vec(comp.mean()));
        mxSetCell(cc, i, from_eigen_mat(comp.covariance()));
        mxSetCell(hc, i, from_eigen_mat(comp.mean_history()));
        mxSetCell(Vc, i, from_eigen_mat(comp.V()));
        pidx[i] = static_cast<double>(comp.init_idx);
        pa[i] = comp.alpha();
        pb[i] = comp.beta();
        pv[i] = comp.v();
        sd = comp.state_dim;
    }
    mxSetField(s, 0, "means", mc);
    mxSetField(s, 0, "covariances", cc);
    mxSetField(s, 0, "weights", from_eigen_vec(mix.weights()));
    mxSetField(s, 0, "dist_type", mxCreateString("TrajectoryGGIW"));
    mxSetField(s, 0, "state_dim", mxCreateDoubleScalar(sd));
    mxSetField(s, 0, "init_indices", idxs);
    mxSetField(s, 0, "mean_histories", hc);
    mxSetField(s, 0, "alphas", alp);
    mxSetField(s, 0, "betas", bet);
    mxSetField(s, 0, "vs", vv);
    mxSetField(s, 0, "Vs", Vc);
    return s;
}

// --- Generic extract dispatch ---

// For each RFS type, get the latest extracted Mixture<T> and convert to MATLAB.
template<typename T, template<typename> class RFS>
static mxArray* extract_from_rfs(multi_target::RFSBase* base) {
    auto& rfs = static_cast<RFS<T>&>(*base);
    auto mix = rfs.extract();
    if (!mix || mix->empty()) {
        const char* f[] = {"means","covariances","weights","dist_type"};
        mxArray* s = mxCreateStructMatrix(1,1,4,f);
        mxSetField(s,0,"means",mxCreateCellMatrix(1,0));
        mxSetField(s,0,"covariances",mxCreateCellMatrix(1,0));
        mxSetField(s,0,"weights",mxCreateDoubleMatrix(1,0,mxREAL));
        mxSetField(s,0,"dist_type",mxCreateString(""));
        return s;
    }
    return mixture_to_matlab(*mix);
}

template<typename T>
static mxArray* extract_dispatch_rfs(multi_target::RFSBase* base, const std::string& rfs_type) {
    if      (rfs_type == "PHD")    return extract_from_rfs<T, multi_target::PHD>(base);
    else if (rfs_type == "CPHD")   return extract_from_rfs<T, multi_target::CPHD>(base);
    else if (rfs_type == "MB")     return extract_from_rfs<T, multi_target::MB>(base);
    else if (rfs_type == "MBM")    return extract_from_rfs<T, multi_target::MBM>(base);
    else if (rfs_type == "LMB")    return extract_from_rfs<T, multi_target::LMB>(base);
    else if (rfs_type == "GLMB")   return extract_from_rfs<T, multi_target::GLMB>(base);
    else if (rfs_type == "JGLMB")  return extract_from_rfs<T, multi_target::JGLMB>(base);
    else if (rfs_type == "PMBM")   return extract_from_rfs<T, multi_target::PMBM>(base);
    else mexErrMsgIdAndTxt("brew:badRFS", "Unknown RFS type: %s", rfs_type.c_str());
    return nullptr;
}

static mxArray* cmd_rfs_extract(uint64_t handle) {
    auto& entry = get_entry(handle);
    auto rfs = get_obj<multi_target::RFSBase>(handle);

    if (entry.dist_type == "Gaussian")
        return extract_dispatch_rfs<models::Gaussian>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "GGIW")
        return extract_dispatch_rfs<models::GGIW>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "TrajectoryGaussian")
        return extract_dispatch_rfs<models::TrajectoryGaussian>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "TrajectoryGGIW")
        return extract_dispatch_rfs<models::TrajectoryGGIW>(rfs.get(), entry.rfs_type);
    else
        mexErrMsgIdAndTxt("brew:badDist", "Unknown dist type: %s", entry.dist_type.c_str());
    return nullptr;
}

// ============================================================
// Birth weight setters
// ============================================================

template<typename T>
static void set_birth_weights_for_rfs(multi_target::RFSBase* base,
                                       const std::string& rfs_type,
                                       const Eigen::VectorXd& weights) {
    if (rfs_type == "PHD") {
        auto& r = static_cast<multi_target::PHD<T>&>(*base);
        r.set_birth_weights(weights);
    } else {
        mexErrMsgIdAndTxt("brew:unsupported",
            "set_birth_weights not supported for RFS type: %s", rfs_type.c_str());
    }
}

static void cmd_rfs_set_birth_weights(uint64_t handle, const Eigen::VectorXd& weights) {
    auto& entry = get_entry(handle);
    auto rfs = get_obj<multi_target::RFSBase>(handle);
    if (entry.dist_type == "Gaussian")
        set_birth_weights_for_rfs<models::Gaussian>(rfs.get(), entry.rfs_type, weights);
    else if (entry.dist_type == "GGIW")
        set_birth_weights_for_rfs<models::GGIW>(rfs.get(), entry.rfs_type, weights);
    else if (entry.dist_type == "TrajectoryGaussian")
        set_birth_weights_for_rfs<models::TrajectoryGaussian>(rfs.get(), entry.rfs_type, weights);
    else if (entry.dist_type == "TrajectoryGGIW")
        set_birth_weights_for_rfs<models::TrajectoryGGIW>(rfs.get(), entry.rfs_type, weights);
    else
        mexErrMsgIdAndTxt("brew:badDist", "Unknown dist type: %s", entry.dist_type.c_str());
}

// ============================================================
// Cardinality getters (CPHD, GLMB, JGLMB)
// ============================================================

template<typename T>
static mxArray* get_cardinality_dispatch(multi_target::RFSBase* base,
                                          const std::string& rfs_type) {
    if (rfs_type == "CPHD") {
        auto& r = static_cast<multi_target::CPHD<T>&>(*base);
        return from_eigen_vec(r.cardinality());
    } else if (rfs_type == "GLMB") {
        auto& r = static_cast<multi_target::GLMB<T>&>(*base);
        return from_eigen_vec(r.cardinality());
    } else if (rfs_type == "JGLMB") {
        auto& r = static_cast<multi_target::JGLMB<T>&>(*base);
        return from_eigen_vec(r.cardinality());
    }
    return mxCreateDoubleMatrix(0, 0, mxREAL);
}

static mxArray* cmd_rfs_get_cardinality(uint64_t handle) {
    auto& entry = get_entry(handle);
    auto rfs = get_obj<multi_target::RFSBase>(handle);
    if (entry.dist_type == "Gaussian")
        return get_cardinality_dispatch<models::Gaussian>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "GGIW")
        return get_cardinality_dispatch<models::GGIW>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "TrajectoryGaussian")
        return get_cardinality_dispatch<models::TrajectoryGaussian>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "TrajectoryGGIW")
        return get_cardinality_dispatch<models::TrajectoryGGIW>(rfs.get(), entry.rfs_type);
    return mxCreateDoubleMatrix(0, 0, mxREAL);
}

// ============================================================
// Track history getters (LMB, MBM, GLMB, JGLMB, PMBM)
// ============================================================

// Convert std::map<int, vector<VectorXd>> to MATLAB struct array
static mxArray* track_histories_to_matlab(
        const std::map<int, std::vector<Eigen::VectorXd>>& histories) {
    const char* fields[] = {"id", "states"};
    size_t n = histories.size();
    mxArray* s = mxCreateStructMatrix(1, n, 2, fields);
    size_t i = 0;
    for (const auto& [id, states] : histories) {
        mxSetField(s, i, "id", mxCreateDoubleScalar(id));
        mxArray* sc = mxCreateCellMatrix(1, states.size());
        for (size_t j = 0; j < states.size(); ++j)
            mxSetCell(sc, j, from_eigen_vec(states[j]));
        mxSetField(s, i, "states", sc);
        ++i;
    }
    return s;
}

template<typename T>
static mxArray* get_track_histories_dispatch(multi_target::RFSBase* base,
                                              const std::string& rfs_type) {
    if (rfs_type == "LMB") {
        auto& r = static_cast<multi_target::LMB<T>&>(*base);
        return track_histories_to_matlab(r.track_histories());
    } else if (rfs_type == "MBM") {
        auto& r = static_cast<multi_target::MBM<T>&>(*base);
        return track_histories_to_matlab(r.track_histories());
    } else if (rfs_type == "GLMB") {
        auto& r = static_cast<multi_target::GLMB<T>&>(*base);
        return track_histories_to_matlab(r.track_histories());
    } else if (rfs_type == "JGLMB") {
        auto& r = static_cast<multi_target::JGLMB<T>&>(*base);
        return track_histories_to_matlab(r.track_histories());
    } else if (rfs_type == "PMBM") {
        auto& r = static_cast<multi_target::PMBM<T>&>(*base);
        return track_histories_to_matlab(r.track_histories());
    }
    return mxCreateStructMatrix(1, 0, 0, nullptr);
}

static mxArray* cmd_rfs_get_track_histories(uint64_t handle) {
    auto& entry = get_entry(handle);
    auto rfs = get_obj<multi_target::RFSBase>(handle);
    if (entry.dist_type == "Gaussian")
        return get_track_histories_dispatch<models::Gaussian>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "GGIW")
        return get_track_histories_dispatch<models::GGIW>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "TrajectoryGaussian")
        return get_track_histories_dispatch<models::TrajectoryGaussian>(rfs.get(), entry.rfs_type);
    else if (entry.dist_type == "TrajectoryGGIW")
        return get_track_histories_dispatch<models::TrajectoryGGIW>(rfs.get(), entry.rfs_type);
    return mxCreateStructMatrix(1, 0, 0, nullptr);
}

// ============================================================
// MEX Entry Point
// ============================================================

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Register cleanup on first call
    static bool initialized = false;
    if (!initialized) {
        mexAtExit(cleanup_all);
        initialized = true;
    }

    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("brew:badCmd", "First argument must be a command string.");

    std::string cmd = get_string(prhs[0]);

    // --- Object creation ---
    if (cmd == "create_dynamics") {
        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "Need dynamics type string.");
        plhs[0] = make_handle(cmd_create_dynamics(get_string(prhs[1])));
    }
    else if (cmd == "create_dbscan") {
        double eps = (nrhs > 1) ? mxGetScalar(prhs[1]) : 1.0;
        int mp     = (nrhs > 2) ? static_cast<int>(mxGetScalar(prhs[2])) : 3;
        plhs[0] = make_handle(cmd_create_dbscan(eps, mp));
    }
    else if (cmd == "create_filter") {
        plhs[0] = make_handle(cmd_create_filter(nrhs, prhs));
    }
    else if (cmd == "create_mixture") {
        plhs[0] = make_handle(cmd_create_mixture(nrhs, prhs));
    }
    else if (cmd == "create_rfs") {
        plhs[0] = make_handle(cmd_create_rfs(nrhs, prhs));
    }

    // --- RFS operations ---
    else if (cmd == "rfs_predict") {
        if (nrhs < 3) mexErrMsgIdAndTxt("brew:badArgs", "rfs_predict: handle, dt");
        cmd_rfs_predict(get_handle(prhs[1]), mxGetScalar(prhs[2]));
    }
    else if (cmd == "rfs_correct") {
        if (nrhs < 3) mexErrMsgIdAndTxt("brew:badArgs", "rfs_correct: handle, measurements");
        cmd_rfs_correct(get_handle(prhs[1]), prhs[2]);
    }
    else if (cmd == "rfs_cleanup") {
        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_cleanup: handle");
        cmd_rfs_cleanup(get_handle(prhs[1]));
    }
    else if (cmd == "rfs_cleanup_and_extract") {
        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_cleanup_and_extract: handle");
        uint64_t h = get_handle(prhs[1]);
        cmd_rfs_cleanup(h);
        plhs[0] = cmd_rfs_extract(h);
    }
    else if (cmd == "rfs_extract") {
        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_extract: handle");
        plhs[0] = cmd_rfs_extract(get_handle(prhs[1]));
    }

    // --- Setters ---
    else if (cmd == "rfs_set_birth_weights") {
        if (nrhs < 3) mexErrMsgIdAndTxt("brew:badArgs", "rfs_set_birth_weights: handle, weights");
        Eigen::VectorXd w = to_eigen_vec(prhs[2]);
        cmd_rfs_set_birth_weights(get_handle(prhs[1]), w);
    }

    // --- Getters ---
    else if (cmd == "rfs_get_cardinality") {
        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_get_cardinality: handle");
        plhs[0] = cmd_rfs_get_cardinality(get_handle(prhs[1]));
    }
    else if (cmd == "rfs_get_track_histories") {
        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_get_track_histories: handle");
        plhs[0] = cmd_rfs_get_track_histories(get_handle(prhs[1]));
    }

    // --- Cleanup ---
    else if (cmd == "destroy") {
        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "destroy: handle");
        g_store.erase(get_handle(prhs[1]));
    }
    else if (cmd == "destroy_all") {
        cleanup_all();
    }

    else {
        mexErrMsgIdAndTxt("brew:badCmd", "Unknown command: %s", cmd.c_str());
    }
}
