cmake_minimum_required(VERSION 3.16)

project(optimsolution LANGUAGES CXX)

# ---------------------------------------------------------------------
# Global C++ settings
# ---------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Do NOT set CMAKE_BUILD_TYPE here (VS is multi-config).
# Use: cmake --build build --config Debug/Release

# ---------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------
add_library(optimcore
    src/problems/rastrigin.cpp
    src/problems/rosenbrock.cpp
    src/problems/potential.cpp
    src/problems/ackley.cpp
    src/problems/sphere.cpp
    src/problems/griewank.cpp
    src/problems/levy.cpp
    src/problems/attractivesector.cpp
    src/problems/bohachevsky1.cpp
    src/problems/bohachevsky2.cpp
    src/problems/bohachevsky3.cpp
    src/problems/branin.cpp
    src/problems/camel.cpp
    src/problems/cigar.cpp
    src/problems/cosinemixture.cpp
    src/problems/differentpowers.cpp
    src/problems/diracproblem.cpp
    src/problems/easom.cpp
    src/problems/ellipsoidal.cpp
    src/problems/equalmaxima.cpp
    src/problems/expotential.cpp
    src/problems/goldstein.cpp
    src/problems/griewankrosenbrock.cpp
    src/problems/hansen.cpp
    src/problems/hartmann3.cpp
    src/problems/hartmann6.cpp
    src/problems/rastrigin2.cpp
    src/problems/rotatedrosenbrock.cpp
    src/problems/shekel5.cpp
    src/problems/shekel7.cpp
    src/problems/shekel10.cpp
    src/problems/shubert.cpp
    src/problems/stepellipsoidal.cpp
    src/problems/test2n.cpp
    src/problems/test30n.cpp
    src/problems/antennaarray.cpp
    src/problems/antennaula.cpp
    src/problems/bifunctionalcatalyst.cpp
    src/problems/bucherastrigin.cpp
    src/problems/cassini.cpp
    src/problems/ded1.cpp
    src/problems/ded2.cpp
    src/problems/eld1.cpp
    src/problems/eld2.cpp
    src/problems/eld3.cpp
    src/problems/eld4.cpp
    src/problems/eld5.cpp
    src/problems/fmsynth.cpp
    src/problems/gallagher101.cpp
    src/problems/gallagher21.cpp
    src/problems/heatexchanger.cpp
    src/problems/himmelblau.cpp
    src/problems/hydrothermal.cpp
    src/problems/ik6dof.cpp
    src/problems/katsuura.cpp
    src/problems/lunacekbirastrigin.cpp
    src/problems/messenger.cpp
    src/problems/michalewicz.cpp
    src/problems/ofdmpower.cpp
    src/problems/polyphase.cpp
    src/problems/portfoliomv.cpp
    src/problems/schaffer.cpp
    src/problems/schwefel.cpp
    src/problems/tandem.cpp
    src/problems/tersoffb.cpp
    src/problems/tersoffc.cpp
    src/problems/tnep.cpp
    src/problems/transmissionpricing.cpp
    src/problems/vibratingplatform.cpp
    src/problems/weierstrass.cpp
    src/problems/wirelesscoverage.cpp
    src/problems/zakharov.cpp
    src/problems/sinusoidal.cpp
    src/problems/gascycle.cpp
    src/problems/gkls.cpp
    src/problems/gkls250.cpp
    src/problems/gkls350.cpp
    src/problems/gkls2100.cpp

    src/methods/arq.cpp
    src/methods/de.cpp
    src/methods/pso.cpp
    src/methods/ga.cpp
    src/methods/aco.cpp
    src/methods/acor.cpp
    src/methods/sa.cpp
    src/methods/woa.cpp
    src/methods/mewoa.cpp
    src/methods/abc.cpp
    src/methods/gwo.cpp
    src/methods/egco.cpp
    src/methods/gao.cpp
    src/methods/ppso.cpp
    src/methods/pde.cpp
    src/methods/pga.cpp
    src/methods/psioa.cpp
    src/methods/sioa.cpp
    src/methods/sao.cpp
    src/methods/psao.cpp
    src/methods/bho.cpp
    src/methods/tridentde.cpp
    src/methods/mlshaderl.cpp
    src/methods/jso.cpp
    src/methods/ea4eig.cpp
    src/methods/ude3.cpp
    src/methods/jde.cpp
    src/methods/sade.cpp
    src/methods/cmaes.cpp
    src/methods/aarq.cpp
    src/methods/arqeig.cpp
    src/methods/fuse.cpp
    src/methods/arqeigrl.cpp
    src/methods/gderl.cpp
    src/methods/garq.cpp
    src/methods/polyde.cpp
    src/methods/rarq.cpp
    src/methods/gde.cpp
    src/methods/arqdp.cpp
    src/methods/hjso.cpp
    src/methods/hde.cpp
    src/methods/clpso.cpp

    src/methods/gd.cpp
    src/methods/nm.cpp
    src/methods/lbfgs.cpp
    src/methods/bfgs.cpp

    src/localsearch.cpp
    src/factory.cpp
    src/options.cpp
    src/stop.cpp
    src/config.cpp
    src/sensitivity.cpp
    src/init.cpp
    src/profiler.cpp
)

target_include_directories(optimcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------
# CLI executable (console subsystem)
# ---------------------------------------------------------------------
add_executable(optimsolution src/main.cpp)
target_link_libraries(optimsolution PRIVATE optimcore)

# ---------------------------------------------------------------------
# GUI (defined by -DCMAKE_PROJECT_INCLUDE=cmake/optimsolution_gui.cmake)
# ---------------------------------------------------------------------
if(COMMAND optimsolution_add_gui)
    optimsolution_add_gui()
endif()
