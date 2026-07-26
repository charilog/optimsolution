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
    src/problems/booth.cpp
    src/problems/beale.cpp
    src/problems/matyas.cpp
    src/problems/mccormick.cpp
    src/problems/colville.cpp
    src/problems/dixonprice.cpp
    src/problems/trid.cpp
    src/problems/powell.cpp
    src/problems/alpine1.cpp
    src/problems/salomon.cpp
    src/problems/whitley.cpp
    src/problems/perm.cpp
    src/problems/cassini1.cpp
    src/problems/sagas.cpp
    src/problems/gtoc1.cpp
    src/problems/rosetta.cpp
    src/problems/stirredtankreactor.cpp
    src/problems/weldedbeam.cpp
    src/problems/speedreducer.cpp
    src/problems/pressurevessel.cpp
    src/problems/springdesign.cpp
    src/problems/cantileverbeam.cpp
    src/problems/threebartruss.cpp
    src/problems/geartrain.cpp
    src/problems/eggholder.cpp
    src/problems/crossintray.cpp
    src/problems/holdertable.cpp
    src/problems/bukinn6.cpp
    src/problems/dropwave.cpp
    src/problems/langermann.cpp
    src/problems/vincent.cpp
    src/problems/fiveunevenpeaktrap.cpp
    src/problems/sinusoidal.cpp
    src/problems/gascycle.cpp
    src/problems/gkls.cpp
    src/problems/gkls250.cpp
    src/problems/gkls350.cpp
    src/problems/gkls2100.cpp
    src/problems/cec2022_zakharov.cpp	
    src/problems/cec2022_rosenbrock.cpp	
    src/problems/cec2022_schafferf7.cpp	
    src/problems/cec2022_noncontinuous_rastrigin.cpp
    src/problems/cec2022_levy.cpp
    src/problems/cec2022_hybrid2.cpp
    src/problems/cec2022_hybrid10.cpp
    src/problems/cec2022_hybrid6.cpp
    src/problems/cec2022_composition1.cpp	
    src/problems/cec2022_composition2.cpp	
    src/problems/cec2022_composition6.cpp
    src/problems/cec2022_composition7.cpp
    src/problems/cec2017_f1.cpp
    src/problems/cec2017_f3.cpp
    src/problems/cec2017_f4.cpp
    src/problems/cec2017_f5.cpp
    src/problems/cec2017_f6.cpp
    src/problems/cec2017_f7.cpp
    src/problems/cec2017_f8.cpp
    src/problems/cec2017_f9.cpp
    src/problems/cec2017_f10.cpp
    src/problems/cec2017_f11.cpp
    src/problems/cec2017_f12.cpp
    src/problems/cec2017_f13.cpp
    src/problems/cec2017_f14.cpp
    src/problems/cec2017_f15.cpp
    src/problems/cec2017_f16.cpp
    src/problems/cec2017_f17.cpp
    src/problems/cec2017_f18.cpp
    src/problems/cec2017_f19.cpp
    src/problems/cec2017_f20.cpp
    src/problems/cec2017_f21.cpp
    src/problems/cec2017_f22.cpp
    src/problems/cec2017_f23.cpp
    src/problems/cec2017_f24.cpp
    src/problems/cec2017_f25.cpp
    src/problems/cec2017_f26.cpp
    src/problems/cec2017_f27.cpp
    src/problems/cec2017_f28.cpp
    src/problems/cec2017_f29.cpp
    src/problems/cec2017_f30.cpp
    src/problems/zdt1.cpp
    src/problems/zdt2.cpp
    src/problems/zdt3.cpp
    src/problems/zdt4.cpp
	
	src/problems/weatherirrigation.cpp	
	src/problems/smartportenergy.cpp		
	src/problems/datacentercooling.cpp	
	src/methods/arq.cpp
    src/methods/de.cpp
    src/methods/pso.cpp
    src/methods/mscso.cpp
    src/methods/emscso.cpp
    src/methods/turbo.cpp
    src/methods/rembo.cpp
    src/methods/ccdg2.cpp
    src/methods/cso.cpp
    src/methods/lmmaes.cpp
    src/methods/vkdcmaes.cpp
    src/methods/rmes.cpp
    src/methods/mmes.cpp
    src/methods/ga.cpp
	src/methods/aco.cpp
    src/methods/acor.cpp
    src/methods/sa.cpp
    src/methods/woa.cpp
    src/methods/mewoa.cpp
    src/methods/rdex.cpp
    src/methods/rde.cpp
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
    src/methods/clpso.cpp
    src/methods/tlbo.cpp
    src/methods/jaya.cpp	
    src/methods/sca.cpp	
    src/methods/fa.cpp		
    src/methods/ba.cpp
    src/methods/hs.cpp
    src/methods/cs.cpp
    src/methods/so.cpp	
    src/methods/gsa.cpp	
    src/methods/alo.cpp
    src/methods/hho.cpp
    src/methods/mfo.cpp
    src/methods/mvo.cpp
    src/methods/sma.cpp
    src/methods/mpa.cpp
    src/methods/eo.cpp
    src/methods/wca.cpp
    src/methods/kh.cpp
    src/methods/hba.cpp
	src/methods/arq2.cpp
	src/methods/ujso.cpp	
	src/methods/mjso.cpp
	src/methods/ude.cpp
	src/methods/sfcde.cpp
	src/methods/awjso.cpp
	src/methods/gahs.cpp	
	src/methods/bwjso.cpp
	src/methods/bjso.cpp
	src/methods/nlshadelbc.cpp
	src/methods/arq3.cpp
	src/methods/hades.cpp
	src/methods/lracmaes.cpp
	src/methods/lmcmaes.cpp
	src/methods/sparq.cpp	
	
    src/methods/gd.cpp
    src/methods/nm.cpp
    src/methods/lbfgs.cpp
    src/methods/bfgs.cpp

    src/methods/nsga2.cpp
    src/methods/moead.cpp
    src/methods/mopso.cpp

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
# OpenMP (optional, used for parallel independent runs)
# ---------------------------------------------------------------------
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(optimcore PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(optimcore PUBLIC OPTIM_HAVE_OPENMP=1)
    message(STATUS "optimsolution: OpenMP found -> parallel runs available (config: [global] parallel_runs / omp_threads)")
else()
    message(STATUS "optimsolution: OpenMP not found -> parallel runs disabled (serial fallback)")
endif()

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
