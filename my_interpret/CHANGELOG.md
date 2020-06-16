# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the versioning is mostly derived from [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.15] - 2019-08-26
### Experimental (WIP)
- Multiclass EBM added. Includes visualization and postprocessing. Currently does not support multiclass pairs.

## [v0.1.14] - 2019-08-20
### Fixed
- Fixed occasional browser crash relating to density graphs.
- Fixed decision trees not displaying in Jupyter notebooks.
### Changed
- Dash components no longer pinned. Upgraded to latest.
- Upgrade from dash-table-experiment to dash-table.
- Numerous renames within native code.
### Experimental (WIP)
- Explanation data methods for PDP, EBM enabled for mli interop.

## [v0.1.13] - 2019-08-14
### Added
- EBM has new parameter 'binning_strategy'. Can now support quantile based binning.
- EBM now gracefully handles many edge cases around data.
- Selenium support added for visual smoke tests.
### Fixed
- Method debug_mode now works in wider environments including WSL.
- Linear models in last version returned the same graphs no matter the selection. Fixed.
### Changed
- Testing requirements now fully separate from default user install.
- Internal EBM class has many renames associated with native codebase. Attribute has been changed to Feature.
- Native codebase has many renames. Diff commits from v0.1.12 to v0.1.13 for more details.
- Dependency gevent lightened to take 1.3.6 or greater. This affects cloud/older Python environments.
- Installation for interpret package should now be 'pip install -U interpret'.
- Removal of skope-rules as a required dependency. User now has to install it manually.
- EBM parameter 'cont_n_bins' renamed to 'max_n_bins'.
### Experimental (WIP)
- Extensions validation method is hardened to ensure blackbox specs are closely met.
- Explanation methods data and visual, require key of form ('mli', key), to access mli interop.

## [v0.1.12] - 2019-08-09
### Fixed
- Fixed EBM bug where 2 features with 1 state are included in the dataset.
- Fixed EBM bug that was causing processing of attributes past an attribute combination with 0 useful attributes to
 fail.

## [v0.1.11] - 2019-08-09
### Added
- C++ testing framework added.
- More granular options for training EBM (not public-facing, added for researchers)
### Fixed
- Improved POSIX compliance for build scripts.
- Failure cases handled better for EBM in both Python/native layer.
- Fixed a bug around dash relating to dependencies.
- Removed dead code around web server for visualization.
### Changed
- For Python setup.py, requirements.txt now used for holding dependencies.
- Directory structure changed for whole repository, in preparation for R support.
- Native code further optimized with compiler flags.
- Consistent scaling for EBM plots across all features.
- For explanation's data method, behavior will be non-standard at key equals -1.
- Testing suite for visual interface added via selenium.
### Experimental (WIP)
- Extension system for blackbox explainers added. Enables other packages to register into interpret.
- Data standardization under way, currently for linear, LIME, SHAP where key equals -1 for data method.

## [v0.1.10] - 2019-07-16
### Fixed
- Fix for duplicated logs.
- EBM now throws exception for multi-class (not supported yet).
- Added requests as dependency.
### Changed
- File requirements.txt renamed to dev-requirements.txt
- Native libraries' names now start with 'lib_' prefix.
- Adjusted return type for debug_mode method to provide logging handler.
- EBM native layer upgraded asserts to use logging.
- EBM native layer hardened for edge case data.
- Adjustments to dev dependencies.
- Method debug_mode defaults log level to INFO.

## [v0.1.9] - 2019-06-14
### Added
- Added method debug_mode in develop module.
- Connected native logging to Python layer.
- Native libraries can now be in release/debug mode.
### Fixed
- Increased system compatibility for C++ code.
### Changed
- Debug related methods expose memory info in human readable form.
- Clean-up of logging levels.
- Various internal C+ fixes.

## [v0.1.8] - 2019-06-07
### Fixed
- Fixed calibration issue with EBM.
- Method show_link fix for anonymous explanation lists.
### Changed
- Method show_link now takes same arguments as show.
- Better error messages with random port allocation.
- More testing for various modules.
- Various internal C+ fixes.

## [v0.1.7] - 2019-06-03
### Added
- Added show_link method. Exposes the URL of show(explanation) as a returned string.
### Fixed
- Fixed shutdown_show_server, can now be called multiple times without failure.
### Changed
- Hardened status_show_server method.
- Testing added for interactive module.
- Removal of extra memory allocation in C++ code for EBM.
- Various internal C++ fixes.

## [v0.1.6] - 2019-05-31
### Added
- Integer indexing for preserve method.
- Public-facing CI build added. Important for pull requests.
### Changed
- Visual-related imports are now loaded when visualize is called for explanations.

## [v0.1.5] - 2019-05-30
### Added
- Added preserve method. Can now save visuals into notebook/file - does not work with decision trees.
- Added status_show_server method. Acts as a check for server reachability.
- Exposed init_show_server method. Can adjust address, base_url, etc.
- Added print_debug_info method in develop module. Important for troubleshooting/bug-reports.
### Fixed
- Various internal C++ fixes.
- Minor clean up on example notebooks.
### Changed
- Additional dependency required: psutil.
- Test refactoring.

## [v0.1.4] - 2019-05-23
### Added
- Root path for show server now has a light monitor page.
- Logging registration can now print to both standard streams and files.
### Fixed
- Error handling for non-existent links fixed for visualization backend.
- In some circumstances, Python process will hang. Resolved with new threading.
### Changed
- Unpinning scipy version, upstream dependencies now compatible with latest.
- Show server is now run by a thread directly, not via executor pools.
- Re-enabled notebook/show tests, new threading resolves hangs.
- Small clean-up of setup.py and Azure pipelines config.

## [v0.1.3] - 2019-05-21
### Added
- Model fit can now support lists of lists as instance data.
- Model fit can now support lists for label data.
### Fixed
- Various internal C++ fixes.
### Changed
- Removed hypothesis as public test dependency.
- C++ logging introduced (no public access).

## [v0.1.2] - 2019-05-17
### Added
- EBM can now disable early stopping with run length set to -1.
- EBM tracking of final episodes per base estimator.
### Fixed
- Pinning scipy, until upstream dependencies are compatible.
### Changed
- Clean-up of EBM logging for training.
- Temporary disable of notebook/show tests until CI environment is fixed.

## [v0.1.1] - 2019-05-16
### Added
- Added server shutdown call for 'show' method.
### Fixed
- Axis titles now included in performance explainer.
- Fixed hang on testing interface.

## [v0.1.0] - 2019-05-14
### Added
- Added port number assignments for 'show' method.
- Includes codebase of v0.0.6.
### Changed
- Native code build scripts hardened.
- Libraries are statically linked where possible.
- Code now conforms to Python Black and its associated flake8.

[v0.1.15]: https://github.com/microsoft/interpret/releases/tag/v0.1.15
[v0.1.14]: https://github.com/microsoft/interpret/releases/tag/v0.1.14
[v0.1.13]: https://github.com/microsoft/interpret/releases/tag/v0.1.13
[v0.1.12]: https://github.com/microsoft/interpret/releases/tag/v0.1.12
[v0.1.11]: https://github.com/microsoft/interpret/releases/tag/v0.1.11
[v0.1.10]: https://github.com/microsoft/interpret/releases/tag/v0.1.10
[v0.1.9]: https://github.com/microsoft/interpret/releases/tag/v0.1.9
[v0.1.8]: https://github.com/microsoft/interpret/releases/tag/v0.1.8
[v0.1.7]: https://github.com/microsoft/interpret/releases/tag/v0.1.7
[v0.1.6]: https://github.com/microsoft/interpret/releases/tag/v0.1.6
[v0.1.5]: https://github.com/microsoft/interpret/releases/tag/v0.1.5
[v0.1.4]: https://github.com/microsoft/interpret/releases/tag/v0.1.4
[v0.1.3]: https://github.com/microsoft/interpret/releases/tag/v0.1.3
[v0.1.2]: https://github.com/microsoft/interpret/releases/tag/v0.1.2
[v0.1.1]: https://github.com/microsoft/interpret/releases/tag/v0.1.1
[v0.1.0]: https://github.com/microsoft/interpret/releases/tag/v0.1.0
