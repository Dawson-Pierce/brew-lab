# Bayesian Recursive Estimation Workspace in MATLAB
----------------------------------------

A MATLAB wrapper for target tracking applications using [BREW](https://github.com/Dawson-Pierce/brew). 

This package holds general dynamics models, different distributions, and filters that work together. 

NOTE: This code is still in development and is in no way promised to perform to any specific standard. 

## Getting Started:

To get started, run the function

```
build_brew()
```

After the compile is done, the wrapper should work as intended. A folder `+BREW` will be created automatically using `generator_files/generate_mex.py`, and then `+utils` holds plotting and sampling methods specific for MATLAB for easy-to-run testing. 

The `tests/` folder should ensure the wrapper is working while showing how to initialize the algorithms. 

## Adding New C++ Classes

When you add a new class to the `brew` C++ library, annotate it with `@mex` comment blocks and run `build_brew`. The generator will automatically create the MEX gateway code and MATLAB wrappers.

### Annotation Reference

Place annotations directly above the class declaration in the C++ header.

**Dynamics** (stored as handle, skeleton MATLAB wrapper):
```cpp
// @mex dynamics
// @mex_name MyDynamics
// @mex_args dims:int, param:double
class MyDynamics : public DynamicsBase { ... };
```

**Models** (value class in MATLAB, used for filter I/O):
```cpp
// @mex model
// @mex_name MyModel
// @mex_fields alpha:scalar, beta:scalar, mean:vec, covariance:mat, v:scalar, V:mat
class MyModel : public BaseSingleModel { ... };
```

**Trajectory models** (wraps a base model with windowed state):
```cpp
// @mex model
// @mex_name TrajectoryMyModel
// @mex_trajectory MyModel
```

**Filters** (handle class, stub predict/correct for subclassing):
```cpp
// @mex filter
// @mex_name MyFilter
// @mex_dist MyModel
// @mex_setters window_size:int, decay:scalar, noise:mat
// @mex_handle_setters icp:IcpBase
class MyFilter : public Filter<models::MyModel> { ... };
```

**RFS filters** (handle class, full predict/correct/cleanup via C++):
```cpp
// @mex rfs
// @mex_name MyRFS
// @mex_params threshold:double:1e-4, max_components:int:100
// @mex_init set_intensity
// @mex_has cardinality, track_histories, birth_weights
// @mex_optional_params poisson_rate:double
template <typename T> class MyRFS : public RFSBase { ... };
```

**Clustering** (handle class):
```cpp
// @mex clustering
// @mex_name MyClustering
// @mex_args epsilon:double, min_pts:int
class MyClustering { ... };
```

**ICP algorithms** (handle class with `align` method):
```cpp
// @mex icp
// @mex_name MyIcp
// @mex_namespace template_matching
// @mex_args inner:clone:IcpBase, template:mat_pc
// @mex_params max_iterations:int:50, tolerance:double:1e-6
class MyIcp : public IcpBase { ... };
```

**Models with handle/matrix constructor fields** (e.g., TemplatePose):
```cpp
// @mex model
// @mex_name MyPose
// @mex_fields mean:vec, covariance:mat, rotation:mat
// @mex_create_mat_fields template_points:PointCloud
// @mex_create_int_vec_fields pos_indices
```

### Field Types

| Type | C++ | MATLAB (mixture creation) | MATLAB (model class) |
|------|-----|--------------------------|---------------------|
| `vec` | `Eigen::VectorXd` | cell of column vectors | property |
| `mat` | `Eigen::MatrixXd` | cell of matrices | property |
| `scalar` | `double` | double array | property |
| `int` | `int` | scalar | property |
| `clone:Type` | `unique_ptr<Type>` (cloned from handle) | object with `.handle_` | - |
| `mat_pc` | `PointCloud` (from matrix) | raw d x N matrix | - |

### What Gets Generated

| Source | Output | Overwrite behavior |
|--------|--------|-------------------|
| `@mex dynamics` | `+BREW/+dynamics/<Name>.m` | Skip if exists |
| `@mex model` | `+BREW/+models/<Name>.m` + `<Name>Mixture.m` | Always |
| `@mex filter` | `+BREW/+filters/<Name>.m` | Always |
| `@mex rfs` | `+BREW/+multi_target/<Name>.m` | Always |
| `@mex clustering` | (MEX command only) | Always |
| `@mex icp` | `+BREW/+template_matching/<Name>.m` | Always |
| All annotations | `generator_files/brew_mex.cpp` | Always |
| All models | `+BREW/+models/Mixture.m` | Always |

## Example 1: 
Gamma Gaussian Inverse Wishart applied to PHD filter for simulated measurements using the GGIW Mixture model. Output of test_PHD_GGIW.m. 

![til](./tests/output/GGIW_PHD_3D.gif)

## Example 2: 
Trajectory Set Theory GM-PHD filter implemented as an alternative to labeled RFS. Output of test_trajectory_gm_phd.m. 

You can see the L-scan window in red, and previous target trajectories in the other colors. The L-scan is performed as an efficient implementation of the trajectory set theory filters. 

![til](./tests/output/gm-trajectory.gif)

## Example 3: 
Same test as Example 1, but with the TST-GGIW. Output of test_PHD_TrajectoryGGIW.m. 

![til](./tests/output/TrajectoryGGIW_PHD_3D.gif)
