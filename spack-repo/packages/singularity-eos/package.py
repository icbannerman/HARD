# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os

import spack

from spack.package import *
from spack.error import SpackError
from spack.directives import directive

@directive("singularity_eos_plugins")
def singularity_eos_plugin(name, spackage, relpath):
    def _execute_register(pkg):
        pkg.plugins[name] = (spackage, relpath)

    return _execute_register


def plugin_validator(pkg_name, variant_name, values):
    if values == ("none",):
        return
    for v in values:
        if v not in SingularityEos.plugins:
            raise SpackError(f"Unknown Singularity-EOS plugin '{v}'")


class SingularityEos(CMakePackage, CudaPackage, ROCmPackage):
    """Singularity-EOS: A collection of closure models and tools useful for
    multiphysics codes."""

    homepage = "https://lanl.github.io/singularity-eos/main/index.html"
    git = "https://github.com/lanl/singularity-eos.git"
    url = "https://github.com/lanl/singularity-eos/archive/refs/tags/release-1.6.1.tar.gz"

    maintainers("rbberger")

    # allow `main` version for development
    version("main", branch="main")
    version("1.9.2", commit="9e7de3eccd610e0654e9a05d673ef7b24d32cf31")
    version("1.9.1", commit="1be18426a2c9c26f969bc14a73b482d5beca0217")

    # build with kokkos, kokkos-kernels for offloading support
    variant("kokkos", default=False, description="Enable kokkos")
    variant(
        "kokkos-kernels", default=False, description="Enable kokkos-kernals for linear algebra"
    )

    # for compatibility with downstream projects
    variant("mpi", default=False, description="Build with MPI support")

    # build converters for sesame, stellarcollapse eos's
    variant(
        "build_extra",
        description="Build converters",
        values=any_combination_of("sesame", "stellarcollapse").with_default("none"),
    )

    # build tests
    variant("tests", default=False, description="Build tests")

    # build the Fortran interface
    variant("fortran", default=True, description="Enable building fortran interface")

    # build the Python bindings
    variant("python", default=False, description="Enable building Python bindings")

    # link to EOSPAC for table reads
    variant("eospac", default=True, description="Pull in EOSPAC")

    # enable/disable HDF5 - used to control upstream `spiner` 
    # configuration 
    variant("hdf5", default=False, description="Use hdf5")

    variant("spiner", default=True, description="Use Spiner")

    variant("closure", default=True, description="Build closure module")
    variant("shared", default=False, description="Build shared libs")
    variant("vandv", default=True, description="Enable V&V EOSs in default Singularity::Variant")

    plugins = {}

    singularity_eos_plugin("dust", "self", "example/plugin")

    variant(
        "plugins",
        multi=True,
        default="none",
        validator=plugin_validator,
        description="list of plugins to build",
        when="@1.9.0:"
    )
    variant("variant", default="default", description="include path used for variant header", when="@1.9.0:")

    # building/testing/docs
    depends_on("cmake@3.19:")
    depends_on("catch2@2.13.7", when="@:1.8.0 +tests")
    depends_on("catch2@3.0.1:", when="@1.9.0: +tests")
    depends_on("python@3:", when="+python")
    depends_on("py-numpy", when="+python+tests")
    depends_on("py-pybind11@2.9.1:", when="+python")

    # linear algebra when not using GPUs
    # TODO we can do LA with +kokkos+kokkos-kernels~cuda,
    # so maybe this should be `when="~kokkos-kernels~cuda"`
    depends_on("eigen@3.3.8:", when="~kokkos-kernels")
    requires("+kokkos-kernels", when="+cuda")
    requires("+kokkos-kernels", when="+rocm")

    # eospac when asked for 
    depends_on("eospac", when="+eospac")

    depends_on("ports-of-call@1.4.2,1.5.2:", when="@:1.7.0")
    depends_on("ports-of-call@1.5.2:", when="@1.7.1:")
    depends_on("ports-of-call@1.6.0:", when="@1.9.0:")
    # request HEAD of main branch
    depends_on("ports-of-call@main", when="@main")
    
    depends_on("spiner +kokkos", when="+kokkos+spiner")
    # tell spiner to use HDF5 
    depends_on("spiner +hdf5", when="+hdf5+spiner")

    depends_on("spiner@1.6.3", when="+spiner")

    depends_on("mpark-variant")
    depends_on(
        "mpark-variant",
        patches=patch(
            "https://raw.githubusercontent.com/lanl/singularity-eos/refs/heads/main/utils/gpu_compatibility.patch",
            sha256="c803670cbd95f9b97458fb4ef403de30229ec81566a5b8e5ccb75ad9d0b22541"
        ),
        when="+cuda",
    )
    depends_on(
        "mpark-variant",
        patches=patch(
            "https://raw.githubusercontent.com/lanl/singularity-eos/refs/heads/main/utils/gpu_compatibility.patch",
            sha256="c803670cbd95f9b97458fb4ef403de30229ec81566a5b8e5ccb75ad9d0b22541",
        ),
        when="+rocm",
    )
    depends_on("binutils@:2.39,2.42:+ld")


    #TODO: do we need kokkos,kokkoskernels the exact same version?
    for _myver,_kver in zip(("@:1.6.2","@1.7.0:"),("@3.2:","@3.3:")):
        depends_on("kokkos" + _kver, when=_myver + '+kokkos')
        depends_on("kokkos-kernels" + _kver, when=_myver + '+kokkos-kernels')

    # set up kokkos offloading dependencies
    for _flag in ("~cuda", "+cuda", "~rocm", "+rocm"):
        depends_on("kokkos" + _flag, when="+kokkos" + _flag)

    for _flag in ("~cuda", "+cuda"):
        depends_on("kokkos-kernels" + _flag, when="+kokkos-kernels" + _flag)

    depends_on("kokkos+pic", when="+kokkos-kernels")

    # specfic specs when using GPU/cuda offloading
    # TODO remove +wrapper for clang builds
    # TODO version guard +cuda_lambda
    depends_on("kokkos +wrapper+cuda_lambda", when="+cuda+kokkos")

    # fix for older spacks
    if spack.version.Version(spack.spack_version) >= spack.version.Version("0.17"):
        depends_on("kokkos-kernels", when="+kokkos-kernels")

    for _flag in list(CudaPackage.cuda_arch_values):
        depends_on("kokkos cuda_arch=" + _flag, when="+cuda+kokkos cuda_arch=" + _flag)
        depends_on("kokkos-kernels cuda_arch=" + _flag, when="+cuda+kokkos cuda_arch=" + _flag)

    for _flag in ROCmPackage.amdgpu_targets:
        depends_on("kokkos amdgpu_target=" + _flag, when="+rocm+kokkos amdgpu_target=" + _flag)

    conflicts("cuda_arch=none", when="+cuda", msg="CUDA architecture is required")
    conflicts("amdgpu_target=none", when="+rocm", msg="ROCm architecture is required")

    # NOTE: we can do depends_on("libfoo cppflags='-fPIC -O2'") for compiler options

    # these are mirrored in the cmake configuration
    conflicts("+cuda", when="~kokkos")
    conflicts("+rocm", when="~kokkos")
    conflicts("+kokkos-kernels", when="~kokkos")
    conflicts("+hdf5", when="~spiner")

    # TODO: @dholliday remove when sg_get_eos not singularity
    conflicts("+fortran", when="~closure")

    # NOTE: these are set so that dependencies in downstream projects share
    # common MPI dependence
    for _flag in ("~mpi", "+mpi"):
        depends_on("hdf5~cxx+hl" + _flag, when="+hdf5" + _flag)
        depends_on("py-h5py" + _flag, when="@:1.6.2 " + _flag)
#        depends_on("kokkos-nvcc-wrapper" + _flag, when="+cuda+kokkos" + _flag)

    patch("true.patch")


    # TODO some options are now version specific. For now it should be 
    # benign, but good practice to do some version guards.
    def cmake_args(self):
        args = [
            self.define("SINGULARITY_PATCH_MPARK_VARIANT", False),
            self.define_from_variant("SINGULARITY_USE_CUDA", "cuda"),
            self.define_from_variant("SINGULARITY_USE_KOKKOS", "kokkos"),
            self.define_from_variant("SINGULARITY_USE_KOKKOSKERNELS", "kokkos-kernels"),
            self.define_from_variant("SINGULARITY_USE_FORTRAN", "fortran"),
            self.define_from_variant("SINGULARITY_BUILD_CLOSURE", "closure"),
            self.define_from_variant("SINGULARITY_BUILD_PYTHON", "python"),
            self.define_from_variant("SINGULARITY_USE_SPINER", "spiner"),
            self.define_from_variant("SINGULARITY_USE_SPINER_WITH_HDF5", "hdf5"),
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            self.define_from_variant("SINGULARITY_USE_V_AND_V_EOS", "vandv"),
            self.define("SINGULARITY_BUILD_TESTS", self.run_tests),
            self.define(
                "SINGULARITY_BUILD_SESAME2SPINER", "sesame" in self.spec.variants["build_extra"].value
            ),
            self.define(
                "SINGULARITY_TEST_SESAME",
                ("sesame" in self.spec.variants["build_extra"].value and self.run_tests),
            ),
            self.define(
                "SINGULARITY_BUILD_STELLARCOLLAPSE2SPINER",
                "stellarcollapse" in self.spec.variants["build_extra"].value,
            ),
            self.define(
                "SINGULARITY_TEST_STELLAR_COLLAPSE",
                ("stellarcollapse" in self.spec.variants["build_extra"].value and self.run_tests),
            ),
            self.define("SINGULARITY_TEST_PYTHON", ("+python" in self.spec and self.run_tests)),
            # TODO: guard for older versions, remove for new versions(1.7<)
            self.define("SINGULARITY_USE_HDF5", "^hdf5" in self.spec),
            self.define("SINGULARITY_USE_EOSPAC", "^eospac" in self.spec),
        ]

        if self.spec.satisfies("@1.9.0:"):
            if "none" not in self.spec.variants["plugins"].value:
                pdirs = []
                for p in self.spec.variants["plugins"].value:
                  spackage, path = self.plugins[p]
                  if spackage == "self":
                      pdirs.append(join_path(self.stage.source_path, path))
                  else:
                      pdirs.append(join_path(self.spec[spackage].prefix, path))
                args.append(self.define("SINGULARITY_PLUGINS", ";".join(pdirs)))

            variant_path = self.spec.variants["variant"].value
            if variant_path != "default":
                parts = os.path.normpath('variant_path').split(os.sep)
                if parts[0] in self.plugins.keys():
                    spackage, path = self.plugins[parts[0]]
                    parts[0] = self.spec[spackage].prefix
                    variant_path = join_path(*parts)
                args.append(self.define("SINGULARITY_VARIANT", variant_path))

        if "+rocm" in self.spec:
            args.append(self.define("CMAKE_CXX_COMPILER", self.spec["hip"].hipcc))
            args.append(self.define("CMAKE_C_COMPILER", self.spec["hip"].hipcc))
        if "+kokkos+cuda" in self.spec:
            args.append(self.define("CMAKE_CXX_COMPILER", self.spec["kokkos"].kokkos_cxx))

        if "+kokkos" in self.spec:
            if "cxxstd" in self.spec["kokkos"].variants:
              cxx_std_variant = "cxxstd" # current spack
            else:
              cxx_std_variant = "std" # older spack
            args.append(self.define("CMAKE_CXX_STANDARD", self.spec["kokkos"].variants[cxx_std_variant].value))

        # goldfiles were downloaded into source folder
        goldfiles = os.path.join(self.stage.source_path, "goldfiles.tar.gz")
        if self.spec.satisfies("+tests build_extra=stellarcollapse") and self.run_tests and os.path.exists(goldfiles):
            args.append(self.define("SINGULARITY_GOLDFILE_URL", f"file://{goldfiles}"))

        return args

    def setup_run_environment(self, env):
        if os.path.isdir(self.prefix.lib64):
            lib_dir = self.prefix.lib64
        else:
            lib_dir = self.prefix.lib

        if "+python" in self.spec:
            python_version = self.spec["python"].version.up_to(2)
            python_inst_dir = join_path(
                lib_dir, "python{0}".format(python_version), "site-packages"
            )
            env.prepend_path("PYTHONPATH", python_inst_dir)
