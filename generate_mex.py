#!/usr/bin/env python3
"""
Generate brew_mex.cpp from @mex-annotated C++ headers.

Scans brew/include/brew/**/*.hpp for @mex comment blocks and emits
the complete MEX gateway source file. Re-run whenever the C++ API changes.

Usage:
    python generate_mex.py
    python generate_mex.py --output brew_mex.cpp --include-root brew/include

Annotation reference
--------------------
Dynamics:
    // @mex dynamics
    // @mex_name SingleIntegrator
    // @mex_args dims:int, beta:double

Models (basic):
    // @mex model
    // @mex_name Gaussian
    // @mex_fields mean:vec, covariance:mat
    // @mex_extract_extra basis:mat          (optional read-only extras)

Models (trajectory wrapper):
    // @mex model
    // @mex_name TrajectoryGaussian
    // @mex_trajectory Gaussian

Filters:
    // @mex filter
    // @mex_name EKF
    // @mex_dist Gaussian
    // @mex_setters temporal_decay:scalar, forgetting_factor:scalar

RFS filters:
    // @mex rfs
    // @mex_name PHD
    // @mex_params prune_threshold:double:1e-4, merge_threshold:double:4.0
    // @mex_init set_intensity
    // @mex_has cardinality, track_histories, birth_weights
    // @mex_optional_params poisson_cardinality:double

Clustering:
    // @mex clustering
    // @mex_name DBSCAN
    // @mex_args epsilon:double, min_pts:int

Field types: vec (Eigen::VectorXd), mat (Eigen::MatrixXd), scalar/double, int
"""

import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field as dfield
from typing import List, Tuple


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Field:
    name: str           # C++ accessor name
    type: str           # vec | mat | scalar | int
    matlab_name: str    # MATLAB struct field name


@dataclass
class DynamicsDef:
    name: str
    header: str
    args: List[Tuple[str, str]]


@dataclass
class ModelDef:
    name: str
    header: str
    cpp_type: str
    fields: List[Field]
    is_trajectory: bool = False
    base_model: str = ""
    extract_extra: List[Field] = dfield(default_factory=list)


@dataclass
class FilterDef:
    name: str
    header: str
    dist: str
    setters: List[Tuple[str, str]]


@dataclass
class RFSDef:
    name: str
    header: str
    params: List[Tuple[str, str, str]]
    init_methods: List[str] = dfield(default_factory=list)
    has_cardinality: bool = False
    has_track_histories: bool = False
    has_birth_weights: bool = False
    optional_params: List[Tuple[str, str]] = dfield(default_factory=list)


@dataclass
class ClusteringDef:
    name: str
    header: str
    args: List[Tuple[str, str]]


# =============================================================================
# Utilities
# =============================================================================

def pluralize(name: str) -> str:
    if name == "mean_history":
        return "mean_histories"
    if name.endswith("sis"):
        return name[:-2] + "es"  # basis -> bases
    if name.endswith("s") or name.endswith("x") or name.endswith("z"):
        return name + "es"
    return name + "s"


def to_snake(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def parse_fields(s: str) -> List[Field]:
    fields = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        name = tokens[0].strip()
        ftype = tokens[1].strip()
        mname = tokens[2].strip() if len(tokens) > 2 else pluralize(name)
        fields.append(Field(name, ftype, mname))
    return fields


def parse_args_str(s: str) -> List[Tuple[str, str]]:
    args = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        args.append((tokens[0].strip(), tokens[1].strip()))
    return args


def parse_params_str(s: str) -> List[Tuple[str, str, str]]:
    params = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(":")
        params.append((tokens[0].strip(), tokens[1].strip(), tokens[2].strip()))
    return params


# =============================================================================
# Header scanning
# =============================================================================

def parse_mex_block(lines: List[str], start: int) -> dict:
    tags = {}
    m = re.search(r"@mex\s+(\w+)", lines[start])
    if m:
        tags["_category"] = m.group(1)
    i = start + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("//"):
            break
        content = line.lstrip("/").strip()
        if not content.startswith("@mex_"):
            break
        m2 = re.match(r"@mex_(\w+)\s+(.*)", content)
        if m2:
            tags[m2.group(1)] = m2.group(2).strip()
        i += 1
    return tags


def scan_headers(root: Path):
    dynamics: List[DynamicsDef] = []
    models_list: List[ModelDef] = []
    filters: List[FilterDef] = []
    rfs_types: List[RFSDef] = []
    clustering: List[ClusteringDef] = []

    for hpp in sorted(root.rglob("*.hpp")):
        text = hpp.read_text(encoding="utf-8")
        header = str(hpp.relative_to(root)).replace("\\", "/")
        lines = text.split("\n")

        for i, line in enumerate(lines):
            if "@mex " in line and "@mex_" not in line:
                tags = parse_mex_block(lines, i)
                cat = tags.get("_category", "")

                if cat == "dynamics":
                    args = parse_args_str(tags.get("args", ""))
                    dynamics.append(DynamicsDef(tags["name"], header, args))

                elif cat == "model":
                    name = tags["name"]
                    if "trajectory" in tags:
                        base = tags["trajectory"]
                        models_list.append(ModelDef(
                            name=name, header=header,
                            cpp_type=f"models::Trajectory<models::{base}>",
                            fields=[], is_trajectory=True, base_model=base,
                        ))
                    else:
                        fields = parse_fields(tags.get("fields", ""))
                        extras = parse_fields(tags.get("extract_extra", ""))
                        models_list.append(ModelDef(
                            name=name, header=header,
                            cpp_type=f"models::{name}",
                            fields=fields, extract_extra=extras,
                        ))

                elif cat == "filter":
                    setters = parse_args_str(tags.get("setters", ""))
                    filters.append(FilterDef(tags["name"], header, tags["dist"], setters))

                elif cat == "rfs":
                    params = parse_params_str(tags.get("params", ""))
                    init_methods = [s.strip() for s in tags.get("init", "").split(",") if s.strip()]
                    has = [s.strip() for s in tags.get("has", "").split(",")]
                    opt_params = parse_args_str(tags.get("optional_params", ""))
                    rfs_types.append(RFSDef(
                        name=tags["name"], header=header, params=params,
                        init_methods=init_methods,
                        has_cardinality="cardinality" in has,
                        has_track_histories="track_histories" in has,
                        has_birth_weights="birth_weights" in has,
                        optional_params=opt_params,
                    ))

                elif cat == "clustering":
                    args = parse_args_str(tags.get("args", ""))
                    clustering.append(ClusteringDef(tags["name"], header, args))

    return dynamics, models_list, filters, rfs_types, clustering


# =============================================================================
# Code generation — static sections
# =============================================================================

OBJECT_STORE = r"""
// ============================================================
// Object Store
// ============================================================

struct ObjEntry {
    std::shared_ptr<void> ptr;
    std::string type;
    std::string dist_type;
    std::string rfs_type;
    int timestep = 0;
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
"""

HELPERS = r"""
// ============================================================
// MATLAB <-> C++ Conversion Helpers
// ============================================================

static std::string get_string(const mxArray* a) {
    if (mxIsChar(a)) {
        char* buf = mxArrayToUTF8String(a);
        std::string s(buf);
        mxFree(buf);
        return s;
    }
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
"""

RFS_OPERATIONS = r"""
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
        if (m == 1 && n > 1) {
            meas = to_eigen_mat(meas_mx);
        } else if (n == 1 && m > 1) {
            Eigen::VectorXd v = to_eigen_vec(meas_mx);
            meas.resize(1, v.size());
            meas.row(0) = v.transpose();
        } else {
            meas = to_eigen_mat(meas_mx);
        }
    }
    rfs->correct(meas);
}

static void cmd_rfs_cleanup(uint64_t handle) {
    auto rfs = get_obj<multi_target::RFSBase>(handle);
    rfs->cleanup();
}
"""

TRACK_HISTORY_HELPER = r"""
// ============================================================
// Track history helper
// ============================================================

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
"""


# =============================================================================
# Code generation — dynamic sections
# =============================================================================

def gen_includes(dynamics, models_list, filters, rfs_types, clustering):
    lines = [
        "// brew_mex.cpp — Auto-generated MEX gateway. DO NOT EDIT BY HAND.",
        "// Re-generate with: python generate_mex.py",
        "//",
        '// Usage:  handle = brew_mex(\'command\', args...)',
        "",
        '#include "mex.h"',
        "",
        "#include <map>",
        "#include <memory>",
        "#include <string>",
        "#include <vector>",
        "#include <cstring>",
        "#include <cmath>",
        "#include <functional>",
        "",
        "#include <Eigen/Dense>",
        "",
        "// brew headers (auto-generated from @mex annotations)",
    ]
    headers = set()
    for d in dynamics:
        headers.add(d.header)
    for m in models_list:
        headers.add(m.header)
    for f in filters:
        headers.add(f.header)
    for r in rfs_types:
        headers.add(r.header)
    for c in clustering:
        headers.add(c.header)
    # Always include mixture and bernoulli
    headers.add("brew/models/mixture.hpp")
    headers.add("brew/models/bernoulli.hpp")

    for h in sorted(headers):
        lines.append(f'#include "{h}"')

    lines.append("")
    lines.append("using namespace brew;")
    return "\n".join(lines)


def gen_dynamics_creation(dynamics):
    lines = [
        "",
        "// ============================================================",
        "// Dynamics Creation",
        "// ============================================================",
        "",
        "static uint64_t cmd_create_dynamics(int nrhs, const mxArray* prhs[]) {",
        '    std::string type = get_string(prhs[1]);',
        "    std::shared_ptr<dynamics::DynamicsBase> dyn;",
        "",
    ]
    for i, d in enumerate(dynamics):
        kw = "if" if i == 0 else "else if"
        if not d.args:
            lines.append(f'    {kw} (type == "{d.name}")')
            lines.append(f"        dyn = std::make_shared<dynamics::{d.name}>();")
        else:
            lines.append(f'    {kw} (type == "{d.name}") {{')
            for j, (aname, atype) in enumerate(d.args):
                idx = 2 + j
                if atype == "int":
                    lines.append(f"        int {aname} = (nrhs > {idx}) ? static_cast<int>(mxGetScalar(prhs[{idx}])) : 1;")
                else:
                    lines.append(f"        double {aname} = (nrhs > {idx}) ? mxGetScalar(prhs[{idx}]) : 0.0;")
            astr = ", ".join(a for a, _ in d.args)
            lines.append(f"        dyn = std::make_shared<dynamics::{d.name}>({astr});")
            lines.append("    }")
    lines.append("    else")
    lines.append('        mexErrMsgIdAndTxt("brew:badDyn", "Unknown dynamics type: %s", type.c_str());')
    lines.append("    return store_obj(dyn, type);")
    lines.append("}")
    return "\n".join(lines)


def gen_clustering_creation(clustering):
    lines = [
        "",
        "// ============================================================",
        "// Clustering Creation",
        "// ============================================================",
    ]
    for c in clustering:
        sn = to_snake(c.name)
        lines.append("")
        lines.append(f"static uint64_t cmd_create_{sn}(int nrhs, const mxArray* prhs[]) {{")
        for j, (aname, atype) in enumerate(c.args):
            idx = 1 + j
            dflt = "3" if atype == "int" else "1.0"
            cast = "static_cast<int>" if atype == "int" else ""
            if atype == "int":
                lines.append(f"    int {aname} = (nrhs > {idx}) ? static_cast<int>(mxGetScalar(prhs[{idx}])) : {dflt};")
            else:
                lines.append(f"    double {aname} = (nrhs > {idx}) ? mxGetScalar(prhs[{idx}]) : {dflt};")
        astr = ", ".join(a for a, _ in c.args)
        lines.append(f"    auto obj = std::make_shared<clustering::{c.name}>({astr});")
        lines.append(f'    return store_obj(obj, "{c.name}");')
        lines.append("}")
    return "\n".join(lines)


def gen_filter_creation(filters_list):
    lines = [
        "",
        "// ============================================================",
        "// Filter Creation",
        "// ============================================================",
        "",
        "static uint64_t cmd_create_filter(int nrhs, const mxArray* prhs[]) {",
        "    if (nrhs < 6)",
        '        mexErrMsgIdAndTxt("brew:badArgs", "create_filter needs: type, dyn_handle, Q, H, R");',
        "",
        "    std::string ftype = get_string(prhs[1]);",
        "    uint64_t dyn_h = get_handle(prhs[2]);",
        "    Eigen::MatrixXd Q = to_eigen_mat(prhs[3]);",
        "    Eigen::MatrixXd H = to_eigen_mat(prhs[4]);",
        "    Eigen::MatrixXd R = to_eigen_mat(prhs[5]);",
        "",
        "    auto dyn = get_obj<dynamics::DynamicsBase>(dyn_h);",
        "",
    ]
    for i, f in enumerate(filters_list):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (ftype == "{f.name}") {{')
        lines.append(f"        auto f = std::make_shared<filters::{f.name}>();")
        lines.append("        f->set_dynamics(dyn);")
        lines.append("        f->set_process_noise(Q);")
        lines.append("        f->set_measurement_jacobian(H);")
        lines.append("        f->set_measurement_noise(R);")
        for j, (sname, stype) in enumerate(f.setters):
            idx = 6 + j
            if stype == "int":
                lines.append(f"        if (nrhs > {idx}) f->set_{sname}(static_cast<int>(mxGetScalar(prhs[{idx}])));")
            else:
                lines.append(f"        if (nrhs > {idx}) f->set_{sname}(mxGetScalar(prhs[{idx}]));")
        lines.append(f'        return store_obj(f, "{f.name}", "{f.dist}");')
        lines.append("    }")
    lines.append("    else {")
    lines.append('        mexErrMsgIdAndTxt("brew:badFilter", "Unknown filter type: %s", ftype.c_str());')
    lines.append("        return 0;")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)


# ---------- Mixture creation ----------

def _gen_basic_mixture_create(model):
    fn = f"create_{to_snake(model.name)}_mixture"
    lines = []
    lines.append(f"static uint64_t {fn}(int nrhs, const mxArray* prhs[]) {{")

    # Map prhs indices to fields: [1]=dist_type, [2..]=fields, last=weights
    idx = 2
    decls = []
    for f in model.fields:
        if f.type in ("vec", "mat"):
            decls.append(f"    const mxArray* {f.name}_cell = prhs[{idx}];")
        elif f.type == "scalar":
            decls.append(f"    const mxArray* {f.name}_arr = prhs[{idx}];")
        idx += 1
    weight_idx = idx
    n_required = weight_idx + 1

    lines.append(f"    if (nrhs < {n_required}) mexErrMsgIdAndTxt(\"brew:badArgs\",")
    lines.append(f'        "{model.name} mixture: wrong number of arguments");')
    lines.extend(decls)
    lines.append(f"    const mxArray* weights = prhs[{weight_idx}];")

    # Count and weight pointer
    first_cell = next((f for f in model.fields if f.type in ("vec", "mat")), None)
    if first_cell:
        lines.append(f"    size_t n = mxGetNumberOfElements({first_cell.name}_cell);")
    else:
        lines.append("    size_t n = mxGetNumberOfElements(weights);")
    lines.append("    double* w = mxGetDoubles(weights);")

    for f in model.fields:
        if f.type == "scalar":
            lines.append(f"    double* p_{f.name} = mxGetDoubles({f.name}_arr);")

    mix_type = f"models::Mixture<{model.cpp_type}>"
    lines.append("")
    lines.append(f"    auto mix = std::make_shared<{mix_type}>();")
    lines.append("    for (size_t i = 0; i < n; ++i) {")

    ctor_args = []
    for f in model.fields:
        if f.type == "vec":
            lines.append(f"        Eigen::VectorXd {f.name} = to_eigen_vec(mxGetCell({f.name}_cell, i));")
            ctor_args.append(f.name)
        elif f.type == "mat":
            lines.append(f"        Eigen::MatrixXd {f.name} = to_eigen_mat(mxGetCell({f.name}_cell, i));")
            ctor_args.append(f.name)
        elif f.type == "scalar":
            ctor_args.append(f"p_{f.name}[i]")

    args_str = ", ".join(ctor_args)
    lines.append(f"        mix->add_component(")
    lines.append(f"            std::make_unique<{model.cpp_type}>({args_str}), w[i]);")
    lines.append("    }")
    lines.append(f'    return store_obj(mix, "Mixture", "{model.name}");')
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _gen_trajectory_mixture_create(model, model_map):
    base = model_map[model.base_model]
    fn = f"create_{to_snake(model.name)}_mixture"
    lines = []
    lines.append(f"static uint64_t {fn}(int nrhs, const mxArray* prhs[]) {{")

    idx = 2
    decls = []
    for f in base.fields:
        if f.type in ("vec", "mat"):
            decls.append(f"    const mxArray* {f.name}_cell = prhs[{idx}];")
        elif f.type == "scalar":
            decls.append(f"    const mxArray* {f.name}_arr = prhs[{idx}];")
        idx += 1
    weight_idx = idx
    n_required = weight_idx + 1

    lines.append(f"    if (nrhs < {n_required}) mexErrMsgIdAndTxt(\"brew:badArgs\",")
    lines.append(f'        "{model.name} mixture: wrong number of arguments");')
    lines.extend(decls)
    lines.append(f"    const mxArray* weights = prhs[{weight_idx}];")

    first_cell = next((f for f in base.fields if f.type in ("vec", "mat")), None)
    if first_cell:
        lines.append(f"    size_t n = mxGetNumberOfElements({first_cell.name}_cell);")
    else:
        lines.append("    size_t n = mxGetNumberOfElements(weights);")
    lines.append("    double* w = mxGetDoubles(weights);")

    for f in base.fields:
        if f.type == "scalar":
            lines.append(f"    double* p_{f.name} = mxGetDoubles({f.name}_arr);")

    mix_type = f"models::Mixture<{model.cpp_type}>"
    lines.append("")
    lines.append(f"    auto mix = std::make_shared<{mix_type}>();")
    lines.append("    for (size_t i = 0; i < n; ++i) {")

    base_ctor = []
    for f in base.fields:
        if f.type == "vec":
            lines.append(f"        Eigen::VectorXd {f.name} = to_eigen_vec(mxGetCell({f.name}_cell, i));")
            base_ctor.append(f.name)
        elif f.type == "mat":
            lines.append(f"        Eigen::MatrixXd {f.name} = to_eigen_mat(mxGetCell({f.name}_cell, i));")
            base_ctor.append(f.name)
        elif f.type == "scalar":
            base_ctor.append(f"p_{f.name}[i]")

    lines.append("        int state_dim = static_cast<int>(mean.size());")
    base_args = ", ".join(base_ctor)
    lines.append(f"        {base.cpp_type} base({base_args});")
    lines.append(f"        mix->add_component(")
    lines.append(f"            std::make_unique<{model.cpp_type}>(state_dim, std::move(base)), w[i]);")
    lines.append("    }")
    lines.append(f'    return store_obj(mix, "Mixture", "{model.name}");')
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def gen_mixture_creation(models_list, model_map):
    lines = [
        "",
        "// ============================================================",
        "// Mixture Creation",
        "// ============================================================",
        "",
    ]
    for m in models_list:
        if m.is_trajectory:
            lines.append(_gen_trajectory_mixture_create(m, model_map))
        else:
            lines.append(_gen_basic_mixture_create(m))

    # Dispatcher
    lines.append("static uint64_t cmd_create_mixture(int nrhs, const mxArray* prhs[]) {")
    lines.append("    std::string dt = get_string(prhs[1]);")
    for i, m in enumerate(models_list):
        kw = "if" if i == 0 else "else if"
        fn = f"create_{to_snake(m.name)}_mixture"
        lines.append(f'    {kw} (dt == "{m.name}") return {fn}(nrhs, prhs);')
    lines.append('    else mexErrMsgIdAndTxt("brew:badDist", "Unknown dist type: %s", dt.c_str());')
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines)


# ---------- RFS creation ----------

def gen_rfs_creation(rfs_types, models_list):
    lines = [
        "",
        "// ============================================================",
        "// RFS Filter Creation",
        "// ============================================================",
        "",
        "static void configure_rfs_base(multi_target::RFSBase& rfs, const mxArray* p) {",
        '    rfs.set_prob_detection(get_field_double(p, "prob_detection", 0.9));',
        '    rfs.set_prob_survive(get_field_double(p, "prob_survive", 0.99));',
        '    rfs.set_clutter_rate(get_field_double(p, "clutter_rate", 0.0));',
        '    rfs.set_clutter_density(get_field_double(p, "clutter_density", 0.0));',
        "}",
    ]

    # Per-RFS-type template creation functions
    for rfs in rfs_types:
        sn = to_snake(rfs.name)
        lines.append("")
        lines.append("template<typename T>")
        lines.append(f"static uint64_t create_{sn}_impl(uint64_t filter_h, uint64_t birth_h, const mxArray* p) {{")
        lines.append(f"    auto rfs = std::make_shared<multi_target::{rfs.name}<T>>();")
        lines.append("    configure_rfs_base(*rfs, p);")

        for method in rfs.init_methods:
            lines.append(f"    rfs->{method}(std::make_unique<models::Mixture<T>>());")

        lines.append("")
        lines.append("    auto filt = get_obj<filters::Filter<T>>(filter_h);")
        lines.append("    rfs->set_filter(filt->clone());")
        lines.append("    auto birth = get_obj<models::Mixture<T>>(birth_h);")
        lines.append("    rfs->set_birth_model(birth->clone());")
        lines.append("")

        for pname, ptype, pdefault in rfs.params:
            if ptype == "int":
                lines.append(f'    rfs->set_{pname}(get_field_int(p, "{pname}", {pdefault}));')
            else:
                lines.append(f'    rfs->set_{pname}(get_field_double(p, "{pname}", {pdefault}));')

        for opname, optype in rfs.optional_params:
            lines.append("")
            lines.append(f'    double {opname}_val = get_field_double(p, "{opname}", -1);')
            lines.append(f"    if ({opname}_val > 0) rfs->set_{opname}({opname}_val);")

        lines.append("")
        lines.append(f'    return store_obj(rfs, "{rfs.name}", get_entry(filter_h).dist_type, "{rfs.name}");')
        lines.append("}")

    # Dispatch: rfs_type string → create function (for a given T)
    lines.append("")
    lines.append("template<typename T>")
    lines.append("static uint64_t create_rfs_dispatch_dist(const std::string& rfs_type,")
    lines.append("                                          uint64_t fh, uint64_t bh, const mxArray* p) {")
    for i, rfs in enumerate(rfs_types):
        kw = "if" if i == 0 else "else if"
        sn = to_snake(rfs.name)
        lines.append(f'    {kw} (rfs_type == "{rfs.name}") return create_{sn}_impl<T>(fh, bh, p);')
    lines.append('    else mexErrMsgIdAndTxt("brew:badRFS", "Unknown RFS type: %s", rfs_type.c_str());')
    lines.append("    return 0;")
    lines.append("}")

    # Main dispatcher: dist_type → T
    lines.append("")
    lines.append("static uint64_t cmd_create_rfs(int nrhs, const mxArray* prhs[]) {")
    lines.append("    if (nrhs < 6)")
    lines.append('        mexErrMsgIdAndTxt("brew:badArgs",')
    lines.append('            "create_rfs needs: rfs_type, dist_type, filter_h, birth_h, params");')
    lines.append("    std::string rfs_type  = get_string(prhs[1]);")
    lines.append("    std::string dist_type = get_string(prhs[2]);")
    lines.append("    uint64_t fh = get_handle(prhs[3]);")
    lines.append("    uint64_t bh = get_handle(prhs[4]);")
    lines.append("    const mxArray* params = prhs[5];")
    lines.append("")
    for i, m in enumerate(models_list):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (dist_type == "{m.name}")')
        lines.append(f"        return create_rfs_dispatch_dist<{m.cpp_type}>(rfs_type, fh, bh, params);")
    lines.append("    else")
    lines.append('        mexErrMsgIdAndTxt("brew:badDist", "Unknown dist type: %s", dist_type.c_str());')
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines)


# ---------- Mixture extraction ----------

def _gen_basic_extraction(model):
    all_fields = model.fields + model.extract_extra
    struct_names = [f.matlab_name for f in all_fields] + ["weights", "dist_type"]
    n_fields = len(struct_names)
    fstr = ", ".join(f'"{s}"' for s in struct_names)

    lines = []
    lines.append(f"static mxArray* mixture_to_matlab(const models::Mixture<{model.cpp_type}>& mix) {{")
    lines.append("    size_t n = mix.size();")
    lines.append(f"    const char* fields[] = {{{fstr}}};")
    lines.append(f"    mxArray* s = mxCreateStructMatrix(1, 1, {n_fields}, fields);")
    lines.append("")

    for f in all_fields:
        if f.type in ("vec", "mat"):
            lines.append(f"    mxArray* {f.name}_c = mxCreateCellMatrix(1, n);")
        elif f.type == "scalar":
            lines.append(f"    mxArray* {f.name}_a = mxCreateDoubleMatrix(1, n, mxREAL);")
            lines.append(f"    double* p_{f.name} = mxGetDoubles({f.name}_a);")

    lines.append("")
    lines.append("    for (size_t i = 0; i < n; ++i) {")
    lines.append("        const auto& comp = mix.component(i);")
    for f in all_fields:
        if f.type == "vec":
            lines.append(f"        mxSetCell({f.name}_c, i, from_eigen_vec(comp.{f.name}()));")
        elif f.type == "mat":
            lines.append(f"        mxSetCell({f.name}_c, i, from_eigen_mat(comp.{f.name}()));")
        elif f.type == "scalar":
            lines.append(f"        p_{f.name}[i] = comp.{f.name}();")
    lines.append("    }")
    lines.append("")

    for f in all_fields:
        if f.type in ("vec", "mat"):
            lines.append(f'    mxSetField(s, 0, "{f.matlab_name}", {f.name}_c);')
        elif f.type == "scalar":
            lines.append(f'    mxSetField(s, 0, "{f.matlab_name}", {f.name}_a);')
    lines.append(f'    mxSetField(s, 0, "weights", from_eigen_vec(mix.weights()));')
    lines.append(f'    mxSetField(s, 0, "dist_type", mxCreateString("{model.name}"));')
    lines.append("    return s;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _gen_trajectory_extraction(model, model_map):
    base = model_map[model.base_model]
    extra_base = [f for f in base.fields if f.name not in ("mean", "covariance")]
    extra_extract = base.extract_extra

    struct_names = ["means", "covariances", "weights", "dist_type", "state_dim", "mean_histories"]
    for f in extra_base:
        struct_names.append(f.matlab_name)
    for f in extra_extract:
        struct_names.append(f.matlab_name)
    n_fields = len(struct_names)
    fstr = ", ".join(f'"{s}"' for s in struct_names)

    lines = []
    lines.append(f"static mxArray* mixture_to_matlab(")
    lines.append(f"        const models::Mixture<{model.cpp_type}>& mix) {{")
    lines.append("    size_t n = mix.size();")
    lines.append(f"    const char* fields[] = {{{fstr}}};")
    lines.append(f"    mxArray* s = mxCreateStructMatrix(1, 1, {n_fields}, fields);")
    lines.append("")
    lines.append("    mxArray* mean_c = mxCreateCellMatrix(1, n);")
    lines.append("    mxArray* cov_c = mxCreateCellMatrix(1, n);")
    lines.append("    mxArray* hist_c = mxCreateCellMatrix(1, n);")

    for f in extra_base + extra_extract:
        if f.type in ("vec", "mat"):
            lines.append(f"    mxArray* {f.name}_c = mxCreateCellMatrix(1, n);")
        elif f.type == "scalar":
            lines.append(f"    mxArray* {f.name}_a = mxCreateDoubleMatrix(1, n, mxREAL);")
            lines.append(f"    double* p_{f.name} = mxGetDoubles({f.name}_a);")

    lines.append("    int sd = 0;")
    lines.append("")
    lines.append("    for (size_t i = 0; i < n; ++i) {")
    lines.append("        const auto& comp = mix.component(i);")
    lines.append("        mxSetCell(mean_c, i, from_eigen_vec(comp.mean()));")
    lines.append("        mxSetCell(cov_c, i, from_eigen_mat(comp.covariance()));")
    lines.append("        mxSetCell(hist_c, i, from_eigen_mat(comp.mean_history()));")

    for f in extra_base + extra_extract:
        if f.type == "vec":
            lines.append(f"        mxSetCell({f.name}_c, i, from_eigen_vec(comp.current().{f.name}()));")
        elif f.type == "mat":
            lines.append(f"        mxSetCell({f.name}_c, i, from_eigen_mat(comp.current().{f.name}()));")
        elif f.type == "scalar":
            lines.append(f"        p_{f.name}[i] = comp.current().{f.name}();")

    lines.append("        sd = comp.state_dim;")
    lines.append("    }")
    lines.append("")
    lines.append('    mxSetField(s, 0, "means", mean_c);')
    lines.append('    mxSetField(s, 0, "covariances", cov_c);')
    lines.append('    mxSetField(s, 0, "weights", from_eigen_vec(mix.weights()));')
    lines.append(f'    mxSetField(s, 0, "dist_type", mxCreateString("{model.name}"));')
    lines.append('    mxSetField(s, 0, "state_dim", mxCreateDoubleScalar(sd));')
    lines.append('    mxSetField(s, 0, "mean_histories", hist_c);')

    for f in extra_base + extra_extract:
        if f.type in ("vec", "mat"):
            lines.append(f'    mxSetField(s, 0, "{f.matlab_name}", {f.name}_c);')
        elif f.type == "scalar":
            lines.append(f'    mxSetField(s, 0, "{f.matlab_name}", {f.name}_a);')

    lines.append("    return s;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def gen_extraction(models_list, model_map, rfs_types):
    lines = [
        "",
        "// ============================================================",
        "// Mixture Extraction",
        "// ============================================================",
        "",
    ]
    for m in models_list:
        if m.is_trajectory:
            lines.append(_gen_trajectory_extraction(m, model_map))
        else:
            lines.append(_gen_basic_extraction(m))

    # Generic extract_from_rfs template
    lines.append("template<typename T, template<typename> class RFS>")
    lines.append("static mxArray* extract_from_rfs(multi_target::RFSBase* base) {")
    lines.append("    auto& rfs = static_cast<RFS<T>&>(*base);")
    lines.append("    auto mix = rfs.extract();")
    lines.append("    if (!mix || mix->empty()) {")
    lines.append('        const char* f[] = {"means","covariances","weights","dist_type"};')
    lines.append("        mxArray* s = mxCreateStructMatrix(1,1,4,f);")
    lines.append('        mxSetField(s,0,"means",mxCreateCellMatrix(1,0));')
    lines.append('        mxSetField(s,0,"covariances",mxCreateCellMatrix(1,0));')
    lines.append('        mxSetField(s,0,"weights",mxCreateDoubleMatrix(1,0,mxREAL));')
    lines.append('        mxSetField(s,0,"dist_type",mxCreateString(""));')
    lines.append("        return s;")
    lines.append("    }")
    lines.append("    return mixture_to_matlab(*mix);")
    lines.append("}")
    lines.append("")

    # Dispatch on rfs_type for a given T
    lines.append("template<typename T>")
    lines.append("static mxArray* extract_dispatch_rfs(multi_target::RFSBase* base, const std::string& rfs_type) {")
    for i, rfs in enumerate(rfs_types):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (rfs_type == "{rfs.name}") return extract_from_rfs<T, multi_target::{rfs.name}>(base);')
    lines.append('    else mexErrMsgIdAndTxt("brew:badRFS", "Unknown RFS type: %s", rfs_type.c_str());')
    lines.append("    return nullptr;")
    lines.append("}")
    lines.append("")

    # Main extract dispatcher on dist_type
    lines.append("static mxArray* cmd_rfs_extract(uint64_t handle) {")
    lines.append("    auto& entry = get_entry(handle);")
    lines.append("    auto rfs = get_obj<multi_target::RFSBase>(handle);")
    lines.append("")
    for i, m in enumerate(models_list):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (entry.dist_type == "{m.name}")')
        lines.append(f"        return extract_dispatch_rfs<{m.cpp_type}>(rfs.get(), entry.rfs_type);")
    lines.append("    else")
    lines.append('        mexErrMsgIdAndTxt("brew:badDist", "Unknown dist type: %s", entry.dist_type.c_str());')
    lines.append("    return nullptr;")
    lines.append("}")
    return "\n".join(lines)


# ---------- Birth weights ----------

def gen_birth_weights(models_list, rfs_types):
    bw_rfs = [r for r in rfs_types if r.has_birth_weights]
    if not bw_rfs:
        return ""

    lines = [
        "",
        "// ============================================================",
        "// Birth Weight Setters",
        "// ============================================================",
        "",
        "template<typename T>",
        "static void set_birth_weights_for_rfs(multi_target::RFSBase* base,",
        "                                       const std::string& rfs_type,",
        "                                       const Eigen::VectorXd& weights) {",
    ]
    for i, rfs in enumerate(bw_rfs):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (rfs_type == "{rfs.name}") {{')
        lines.append(f"        auto& r = static_cast<multi_target::{rfs.name}<T>&>(*base);")
        lines.append("        r.set_birth_weights(weights);")
        lines.append("    }")
    lines.append("    else {")
    lines.append('        mexErrMsgIdAndTxt("brew:unsupported",')
    lines.append('            "set_birth_weights not supported for RFS type: %s", rfs_type.c_str());')
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Dispatch on dist_type
    lines.append("static void cmd_rfs_set_birth_weights(uint64_t handle, const Eigen::VectorXd& weights) {")
    lines.append("    auto& entry = get_entry(handle);")
    lines.append("    auto rfs = get_obj<multi_target::RFSBase>(handle);")
    for i, m in enumerate(models_list):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (entry.dist_type == "{m.name}")')
        lines.append(f"        set_birth_weights_for_rfs<{m.cpp_type}>(rfs.get(), entry.rfs_type, weights);")
    lines.append("    else")
    lines.append('        mexErrMsgIdAndTxt("brew:badDist", "Unknown dist type: %s", entry.dist_type.c_str());')
    lines.append("}")
    return "\n".join(lines)


# ---------- Cardinality ----------

def gen_cardinality(models_list, rfs_types):
    card_rfs = [r for r in rfs_types if r.has_cardinality]
    if not card_rfs:
        return ""

    lines = [
        "",
        "// ============================================================",
        "// Cardinality Getters",
        "// ============================================================",
        "",
        "template<typename T>",
        "static mxArray* get_cardinality_dispatch(multi_target::RFSBase* base,",
        "                                          const std::string& rfs_type) {",
    ]
    for i, rfs in enumerate(card_rfs):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (rfs_type == "{rfs.name}") {{')
        lines.append(f"        auto& r = static_cast<multi_target::{rfs.name}<T>&>(*base);")
        lines.append("        return from_eigen_vec(r.cardinality());")
        lines.append("    }")
    lines.append("    return mxCreateDoubleMatrix(0, 0, mxREAL);")
    lines.append("}")
    lines.append("")

    lines.append("static mxArray* cmd_rfs_get_cardinality(uint64_t handle) {")
    lines.append("    auto& entry = get_entry(handle);")
    lines.append("    auto rfs = get_obj<multi_target::RFSBase>(handle);")
    for i, m in enumerate(models_list):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (entry.dist_type == "{m.name}")')
        lines.append(f"        return get_cardinality_dispatch<{m.cpp_type}>(rfs.get(), entry.rfs_type);")
    lines.append("    return mxCreateDoubleMatrix(0, 0, mxREAL);")
    lines.append("}")
    return "\n".join(lines)


# ---------- Track histories ----------

def gen_track_histories(models_list, rfs_types):
    th_rfs = [r for r in rfs_types if r.has_track_histories]
    if not th_rfs:
        return ""

    lines = [
        "",
        "// ============================================================",
        "// Track History Getters",
        "// ============================================================",
        "",
        "template<typename T>",
        "static mxArray* get_track_histories_dispatch(multi_target::RFSBase* base,",
        "                                              const std::string& rfs_type) {",
    ]
    for i, rfs in enumerate(th_rfs):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (rfs_type == "{rfs.name}") {{')
        lines.append(f"        auto& r = static_cast<multi_target::{rfs.name}<T>&>(*base);")
        lines.append("        return track_histories_to_matlab(r.track_histories());")
        lines.append("    }")
    lines.append("    return mxCreateStructMatrix(1, 0, 0, nullptr);")
    lines.append("}")
    lines.append("")

    lines.append("static mxArray* cmd_rfs_get_track_histories(uint64_t handle) {")
    lines.append("    auto& entry = get_entry(handle);")
    lines.append("    auto rfs = get_obj<multi_target::RFSBase>(handle);")
    for i, m in enumerate(models_list):
        kw = "if" if i == 0 else "else if"
        lines.append(f'    {kw} (entry.dist_type == "{m.name}")')
        lines.append(f"        return get_track_histories_dispatch<{m.cpp_type}>(rfs.get(), entry.rfs_type);")
    lines.append("    return mxCreateStructMatrix(1, 0, 0, nullptr);")
    lines.append("}")
    return "\n".join(lines)


# ---------- mexFunction entry point ----------

def gen_mex_function(clustering):
    lines = [
        "",
        "// ============================================================",
        "// MEX Entry Point",
        "// ============================================================",
        "",
        "void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {",
        "    static bool initialized = false;",
        "    if (!initialized) {",
        "        mexAtExit(cleanup_all);",
        "        initialized = true;",
        "    }",
        "",
        "    if (nrhs < 1 || !mxIsChar(prhs[0]))",
        '        mexErrMsgIdAndTxt("brew:badCmd", "First argument must be a command string.");',
        "",
        "    std::string cmd = get_string(prhs[0]);",
        "",
        "    // --- Object creation ---",
        '    if (cmd == "create_dynamics") {',
        '        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "Need dynamics type string.");',
        "        plhs[0] = make_handle(cmd_create_dynamics(nrhs, prhs));",
        "    }",
    ]

    # Clustering commands
    for c in clustering:
        sn = to_snake(c.name)
        cmd_name = f"create_{sn}"
        lines.append(f'    else if (cmd == "{cmd_name}") {{')
        lines.append(f"        plhs[0] = make_handle(cmd_create_{sn}(nrhs, prhs));")
        lines.append("    }")

    lines.extend([
        '    else if (cmd == "create_filter") {',
        "        plhs[0] = make_handle(cmd_create_filter(nrhs, prhs));",
        "    }",
        '    else if (cmd == "create_mixture") {',
        "        plhs[0] = make_handle(cmd_create_mixture(nrhs, prhs));",
        "    }",
        '    else if (cmd == "create_rfs") {',
        "        plhs[0] = make_handle(cmd_create_rfs(nrhs, prhs));",
        "    }",
        "",
        "    // --- RFS operations ---",
        '    else if (cmd == "rfs_predict") {',
        '        if (nrhs < 3) mexErrMsgIdAndTxt("brew:badArgs", "rfs_predict: handle, dt");',
        "        cmd_rfs_predict(get_handle(prhs[1]), mxGetScalar(prhs[2]));",
        "    }",
        '    else if (cmd == "rfs_correct") {',
        '        if (nrhs < 3) mexErrMsgIdAndTxt("brew:badArgs", "rfs_correct: handle, measurements");',
        "        cmd_rfs_correct(get_handle(prhs[1]), prhs[2]);",
        "    }",
        '    else if (cmd == "rfs_cleanup") {',
        '        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_cleanup: handle");',
        "        cmd_rfs_cleanup(get_handle(prhs[1]));",
        "    }",
        '    else if (cmd == "rfs_cleanup_and_extract") {',
        '        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_cleanup_and_extract: handle");',
        "        uint64_t h = get_handle(prhs[1]);",
        "        cmd_rfs_cleanup(h);",
        "        plhs[0] = cmd_rfs_extract(h);",
        "    }",
        '    else if (cmd == "rfs_extract") {',
        '        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_extract: handle");',
        "        plhs[0] = cmd_rfs_extract(get_handle(prhs[1]));",
        "    }",
        "",
        "    // --- Setters ---",
        '    else if (cmd == "rfs_set_birth_weights") {',
        '        if (nrhs < 3) mexErrMsgIdAndTxt("brew:badArgs", "rfs_set_birth_weights: handle, weights");',
        "        Eigen::VectorXd w = to_eigen_vec(prhs[2]);",
        "        cmd_rfs_set_birth_weights(get_handle(prhs[1]), w);",
        "    }",
        "",
        "    // --- Getters ---",
        '    else if (cmd == "rfs_get_cardinality") {',
        '        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_get_cardinality: handle");',
        "        plhs[0] = cmd_rfs_get_cardinality(get_handle(prhs[1]));",
        "    }",
        '    else if (cmd == "rfs_get_track_histories") {',
        '        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "rfs_get_track_histories: handle");',
        "        plhs[0] = cmd_rfs_get_track_histories(get_handle(prhs[1]));",
        "    }",
        "",
        "    // --- Cleanup ---",
        '    else if (cmd == "destroy") {',
        '        if (nrhs < 2) mexErrMsgIdAndTxt("brew:badArgs", "destroy: handle");',
        "        g_store.erase(get_handle(prhs[1]));",
        "    }",
        '    else if (cmd == "destroy_all") {',
        "        cleanup_all();",
        "    }",
        "",
        "    else {",
        '        mexErrMsgIdAndTxt("brew:badCmd", "Unknown command: %s", cmd.c_str());',
        "    }",
        "}",
    ])
    return "\n".join(lines)


# =============================================================================
# Main assembly — C++ MEX gateway
# =============================================================================

def generate_mex(dynamics, models_list, filters, rfs_types, clustering):
    model_map = {m.name: m for m in models_list}

    sections = [
        gen_includes(dynamics, models_list, filters, rfs_types, clustering),
        OBJECT_STORE,
        HELPERS,
        gen_dynamics_creation(dynamics),
        gen_clustering_creation(clustering),
        gen_filter_creation(filters),
        gen_mixture_creation(models_list, model_map),
        gen_rfs_creation(rfs_types, models_list),
        RFS_OPERATIONS,
        TRACK_HISTORY_HELPER,
        gen_extraction(models_list, model_map, rfs_types),
        gen_birth_weights(models_list, rfs_types),
        gen_cardinality(models_list, rfs_types),
        gen_track_histories(models_list, rfs_types),
        gen_mex_function(clustering),
    ]
    return "\n".join(s for s in sections if s) + "\n"


# =============================================================================
# MATLAB wrapper generation
# =============================================================================

def _write_if_changed(path: Path, content: str, overwrite: bool) -> str:
    """Write file if it doesn't exist or content changed. Returns status string."""
    if path.exists() and not overwrite:
        return f"  skip (exists): {path}"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == content:
            return f"  unchanged:     {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"  wrote:         {path}"


def _matlab_model_data_class(model: ModelDef, model_map: dict) -> str:
    """Generate a +BREW/+models/<Name>.m value class (single distribution)."""
    name = model.name

    if model.is_trajectory:
        base = model_map[model.base_model]
        base_extra = [f for f in base.fields if f.name not in ("mean", "covariance")]
        base_extract = base.extract_extra
        all_props = [
            ("mean", "Stacked state mean"),
            ("covariance", "Stacked state covariance"),
            ("state_dim", "Single-step state dimension"),
            ("mean_history", "state_dim x T matrix of mean history"),
        ]
        for f in base_extra + base_extract:
            all_props.append((f.name, ""))
    else:
        all_fields = model.fields + model.extract_extra
        all_props = [(f.name, "") for f in all_fields]

    props_lines = []
    for pname, pcomment in all_props:
        if pcomment:
            props_lines.append(f"        {pname} = []  % {pcomment}")
        else:
            props_lines.append(f"        {pname} = []")
    props_str = "\n".join(props_lines)

    ctor_args = [p[0] for p in all_props]
    ctor_sig = ", ".join(ctor_args)
    ctor_lines = []
    for i, pname in enumerate(ctor_args):
        ctor_lines.append(f"            if nargin >= {i+1}, obj.{pname} = {pname}; end")
    ctor_str = "\n".join(ctor_lines)

    # fromMixture static method — extract component i from a Mixture
    from_lines = []
    for pname, _ in all_props:
        if pname in ("mean", "covariance"):
            from_lines.append(f"            obj.{pname} = mix.{pname}s{{idx}};")
        elif pname == "state_dim":
            from_lines.append(f"            obj.state_dim = mix.state_dim;")
        elif pname == "mean_history":
            from_lines.append(f"            if ~isempty(mix.mean_histories)")
            from_lines.append(f"                obj.mean_history = mix.mean_histories{{idx}};")
            from_lines.append(f"            end")
        elif pname in ("alpha", "beta", "v"):
            from_lines.append(f"            obj.{pname} = mix.{pname}s(idx);")
        elif pname in ("V", "basis"):
            from_lines.append(f"            if ~isempty(mix.{pluralize(pname)})")
            from_lines.append(f"                obj.{pname} = mix.{pluralize(pname)}{{idx}};")
            from_lines.append(f"            end")
    from_str = "\n".join(from_lines)

    content = f"""\
classdef {name}
%{name} Single-distribution data class (auto-generated).
%   Value class holding {name} distribution parameters.
%   Used as input/output for MATLAB-side filter predict/correct.
%
%   Construction:
%       dist = BREW.models.{name}(mean, covariance, ...);
%
%   Extract from Mixture:
%       dist = BREW.models.{name}.fromMixture(mixture, idx);

    properties
{props_str}
    end

    methods
        function obj = {name}({ctor_sig})
{ctor_str}
        end
    end

    methods (Static)
        function obj = fromMixture(mix, idx)
            %FROMMIXTURE Extract component idx from a BREW.models.Mixture.
            if nargin < 2, idx = 1; end
            obj = BREW.models.{name}();
{from_str}
        end
    end
end
"""
    return content


def _matlab_mixture_class(model: ModelDef, model_map: dict) -> str:
    """Generate a +BREW/+models/<Name>Mixture.m class."""
    name = model.name
    cls = f"{name}Mixture"

    # Determine fields and mex args order
    if model.is_trajectory:
        base = model_map[model.base_model]
        fields = base.fields
    else:
        fields = model.fields

    # Build inputParser parameters and mex call args
    params = []
    mex_args = []
    conversions = []  # lines to convert cell->array for scalars

    for f in fields:
        if f.type in ("vec", "mat"):
            params.append(f"            addParameter(p, '{f.matlab_name}', {{}});")
            mex_args.append(f"p.Results.{f.matlab_name}")
        elif f.type == "scalar":
            params.append(f"            addParameter(p, '{f.matlab_name}', []);")
            var = f"arg_{f.name}"
            conversions.append(
                f"            {var} = p.Results.{f.matlab_name}; "
                f"if iscell({var}), {var} = cell2mat({var}); end")
            mex_args.append(var)

    params.append("            addParameter(p, 'weights', []);")
    mex_args.append("p.Results.weights")

    params_str = "\n".join(params)
    conv_str = "\n".join(conversions)
    args_str = ", ...\n                ".join(mex_args)

    content = f"""\
classdef {cls} < handle
%{cls} Birth mixture wrapper for {name} distributions (auto-generated).

    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "{name}"
    end

    methods
        function obj = {cls}(varargin)
            p = inputParser;
            p.CaseSensitive = true;
{params_str}
            parse(p, varargin{{:}});
{conv_str}
            obj.handle_ = brew_mex('create_mixture', '{name}', ...
                {args_str});
        end

        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
"""
    return content


def _matlab_rfs_class(rfs: RFSDef) -> str:
    """Generate a +BREW/+multi_target/<Name>.m class."""
    name = rfs.name

    # Build inputParser params from rfs base + rfs-specific
    base_params = [
        ("filter", "[]"),
        ("birth_model", "[]"),
        ("prob_detection", "0.9"),
        ("prob_survive", "0.99"),
        ("clutter_rate", "0.0"),
        ("clutter_density", "0.0"),
    ]

    rfs_params = []
    for pname, ptype, pdefault in rfs.params:
        rfs_params.append((pname, pdefault))

    for opname, optype in rfs.optional_params:
        rfs_params.append((opname, "-1"))

    # inputParser lines
    ip_lines = []
    for pname, pdefault in base_params + rfs_params:
        ip_lines.append(f"            addParameter(p, '{pname}', {pdefault});")

    # struct fields (all params except filter and birth_model)
    struct_fields = []
    for pname, _ in base_params[2:]:  # skip filter, birth_model
        struct_fields.append(f"                '{pname}', p.Results.{pname}")
    for pname, _ in rfs_params:
        struct_fields.append(f"                '{pname}', p.Results.{pname}")

    ip_str = "\n".join(ip_lines)
    struct_str = ", ...\n".join(struct_fields)

    # Optional methods
    extra_methods = ""
    if rfs.has_birth_weights:
        extra_methods += """
        function set_birth_weights(obj, weights)
            brew_mex('rfs_set_birth_weights', obj.handle_, weights);
        end
"""
    if rfs.has_cardinality:
        extra_methods += """
        function card = cardinality(obj)
            card = brew_mex('rfs_get_cardinality', obj.handle_);
        end
"""
    if rfs.has_track_histories:
        extra_methods += """
        function h = track_histories(obj)
            h = brew_mex('rfs_get_track_histories', obj.handle_);
        end
"""

    content = f"""\
classdef {name} < handle
%{name} {name} multi-target filter wrapper (auto-generated).

    properties (SetAccess = private)
        handle_    uint64
        dist_type_ string
    end

    methods
        function obj = {name}(varargin)
            p = inputParser;
{ip_str}
            parse(p, varargin{{:}});

            filt  = p.Results.filter;
            birth = p.Results.birth_model;
            obj.dist_type_ = filt.dist_type_;

            params = struct( ...
{struct_str});

            obj.handle_ = brew_mex('create_rfs', '{name}', obj.dist_type_, ...
                filt.handle_, birth.handle_, params);
        end

        function predict(obj, dt)
            brew_mex('rfs_predict', obj.handle_, dt);
        end

        function correct(obj, varargin)
            if nargin == 3, measurements = varargin{{2}};
            else, measurements = varargin{{1}}; end
            if isempty(measurements), measurements = zeros(1, 0); end
            brew_mex('rfs_correct', obj.handle_, measurements);
        end

        function est = cleanup(obj)
            raw = brew_mex('rfs_cleanup_and_extract', obj.handle_);
            est = BREW.models.Mixture(raw, obj.dist_type_);
        end

        function est = extract(obj)
            raw = brew_mex('rfs_extract', obj.handle_);
            est = BREW.models.Mixture(raw, obj.dist_type_);
        end
{extra_methods}
        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
"""
    return content


def _matlab_dynamics_class(dyn: DynamicsDef) -> str:
    """Generate a skeleton +BREW/+dynamics/<Name>.m class."""
    name = dyn.name

    # Constructor args
    if dyn.args:
        ctor_params = []
        ctor_store = []
        mex_args = []
        props = []
        for aname, atype in dyn.args:
            ctor_params.append(aname)
            ctor_store.append(f"            obj.{aname}_ = {aname};")
            mex_args.append(aname)
            props.append(f"        {aname}_")
        ctor_sig = ", ".join(ctor_params)
        ctor_store_str = "\n".join(ctor_store)
        mex_extra = ", " + ", ".join(mex_args)
        props_str = "\n".join(props)
        props_block = f"""
    properties (SetAccess = private)
{props_str}
    end
"""
    else:
        ctor_sig = ""
        ctor_store_str = ""
        mex_extra = ""
        props_block = ""

    content = f"""\
classdef {name} < handle
%{name} Dynamics wrapper (auto-generated skeleton).
%   Add getStateMat, getInputMat, propagateState methods as needed.

    properties (SetAccess = private)
        handle_ uint64
    end

    properties
        M = []  % extent propagation matrix (optional)
    end
{props_block}
    methods
        function obj = {name}({ctor_sig})
{ctor_store_str}
            obj.handle_ = brew_mex('create_dynamics', '{name}'{mex_extra});
        end

        %% TODO: implement dynamics methods for MATLAB-side usage
        % function F = getStateMat(obj, dt, varargin) ... end
        % function G = getInputMat(obj, dt, varargin) ... end
        % function x = propagateState(obj, dt, state, varargin) ... end

        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
"""
    return content


def _matlab_filter_class(filt: FilterDef) -> str:
    """Generate a +BREW/+filters/<Name>.m class (always overwritten)."""
    name = filt.name
    dist = filt.dist

    # Build inputParser params
    base_ip = [
        "            addParameter(p, 'dyn_obj', []);",
        "            addParameter(p, 'process_noise', []);",
        "            addParameter(p, 'H', []);",
        "            addParameter(p, 'measurement_noise', []);",
    ]
    extra_ip = []
    extra_store = []
    extra_props = []
    mex_extra_args = []

    for sname, stype in filt.setters:
        dflt = "50" if sname == "window_size" else "1.0"
        extra_ip.append(f"            addParameter(p, '{sname}', {dflt});")
        extra_store.append(f"            obj.{sname}_ = p.Results.{sname};")
        extra_props.append(f"        {sname}_")
        mex_extra_args.append(f"p.Results.{sname}")

    ip_str = "\n".join(base_ip + extra_ip)
    store_str = "\n".join(extra_store)
    props_str = "\n".join(extra_props) if extra_props else ""
    mex_extra = ""
    if mex_extra_args:
        mex_extra = ", ...\n                " + ", ...\n                ".join(mex_extra_args)

    extra_props_block = ""
    if props_str:
        extra_props_block = f"""
    properties
{props_str}
    end
"""

    content = f"""\
classdef {name} < handle
%{name} Filter wrapper (auto-generated).
%   Constructor creates a C++ filter handle for use with RFS filters.
%   Override predict/correct in a subclass for MATLAB-side single-target usage.
%
%   The predict/correct methods accept and return BREW.models.{dist} objects.

    properties (SetAccess = private)
        handle_ uint64
        dist_type_ string = "{dist}"
    end

    properties
        dyn_obj_
        H_
        process_noise_
        measurement_noise_
    end
{extra_props_block}
    methods
        function obj = {name}(varargin)
            p = inputParser;
{ip_str}
            parse(p, varargin{{:}});
            obj.dyn_obj_ = p.Results.dyn_obj;
            obj.H_ = p.Results.H;
            obj.process_noise_ = p.Results.process_noise;
            obj.measurement_noise_ = p.Results.measurement_noise;
{store_str}
            obj.handle_ = brew_mex('create_filter', '{name}', ...
                p.Results.dyn_obj.handle_, p.Results.process_noise, ...
                p.Results.H, p.Results.measurement_noise{mex_extra});
        end

        function nextDist = predict(obj, dt, prevDist) %#ok<STOUT,INUSD>
            %PREDICT Filter predict step.
            %   prevDist: BREW.models.{dist}
            %   returns:  BREW.models.{dist}
            %
            %   Override in a subclass to implement MATLAB-side prediction.
            error('BREW:notImplemented', ...
                '{name}.predict is not implemented. Subclass {name} and override this method.');
        end

        function [nextDist, likelihood] = correct(obj, meas, prevDist) %#ok<STOUT,INUSD>
            %CORRECT Filter correct step.
            %   meas:     measurement vector or matrix
            %   prevDist: BREW.models.{dist}
            %   returns:  BREW.models.{dist}, scalar likelihood
            %
            %   Override in a subclass to implement MATLAB-side correction.
            error('BREW:notImplemented', ...
                '{name}.correct is not implemented. Subclass {name} and override this method.');
        end

        function delete(obj)
            if obj.handle_ ~= 0
                try brew_mex('destroy', obj.handle_); catch, end
            end
        end
    end
end
"""
    return content


def generate_matlab(dynamics, models_list, filters, rfs_types, clustering, outdir: Path):
    """Generate all MATLAB wrapper classes under the +BREW directory."""
    model_map = {m.name: m for m in models_list}
    messages = []

    # --- Model data classes (always overwrite) ---
    for m in models_list:
        path = outdir / "+models" / f"{m.name}.m"
        content = _matlab_model_data_class(m, model_map)
        messages.append(_write_if_changed(path, content, overwrite=True))

    # --- Model mixture classes for birth models (always overwrite) ---
    for m in models_list:
        cls_name = f"{m.name}Mixture"
        path = outdir / "+models" / f"{cls_name}.m"
        content = _matlab_mixture_class(m, model_map)
        messages.append(_write_if_changed(path, content, overwrite=True))

    # --- RFS filters (always overwrite) ---
    for rfs in rfs_types:
        path = outdir / "+multi_target" / f"{rfs.name}.m"
        content = _matlab_rfs_class(rfs)
        messages.append(_write_if_changed(path, content, overwrite=True))

    # --- Dynamics (skeleton, skip if exists) ---
    for dyn in dynamics:
        path = outdir / "+dynamics" / f"{dyn.name}.m"
        content = _matlab_dynamics_class(dyn)
        messages.append(_write_if_changed(path, content, overwrite=False))

    # --- Filters (always overwrite) ---
    for filt in filters:
        path = outdir / "+filters" / f"{filt.name}.m"
        content = _matlab_filter_class(filt)
        messages.append(_write_if_changed(path, content, overwrite=True))

    return messages


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate brew_mex.cpp and MATLAB wrappers")
    parser.add_argument("--output", "-o", default="brew_mex.cpp",
                        help="Output C++ file (default: brew_mex.cpp)")
    parser.add_argument("--include-root", default="brew/include",
                        help="Root include directory (default: brew/include)")
    parser.add_argument("--matlab-dir", default=None,
                        help="Generate MATLAB wrappers in this directory (e.g., +BREW)")
    args = parser.parse_args()

    root = Path(args.include_root)
    if not root.exists():
        print(f"Error: include root '{root}' not found", file=sys.stderr)
        sys.exit(1)

    dynamics, models_list, filters, rfs_types, clustering = scan_headers(root)

    print(f"Found: {len(dynamics)} dynamics, {len(models_list)} models, "
          f"{len(filters)} filters, {len(rfs_types)} RFS types, {len(clustering)} clustering")

    # Generate C++ MEX gateway
    code = generate_mex(dynamics, models_list, filters, rfs_types, clustering)
    Path(args.output).write_text(code, encoding="utf-8")
    print(f"Generated {args.output} ({len(code)} bytes)")

    # Generate MATLAB wrappers
    if args.matlab_dir:
        outdir = Path(args.matlab_dir)
        messages = generate_matlab(
            dynamics, models_list, filters, rfs_types, clustering, outdir)
        print(f"\nMATLAB wrappers ({outdir}):")
        for msg in messages:
            print(msg)


if __name__ == "__main__":
    main()
