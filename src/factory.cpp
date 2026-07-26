#include "factory.h"
#include "utils.h"

// problems
#include "problems/ackley.h"
#include "problems/attractivesector.h"
#include "problems/bohachevsky1.h"
#include "problems/bohachevsky2.h"
#include "problems/bohachevsky3.h"
#include "problems/branin.h"
#include "problems/camel.h"
#include "problems/cigar.h"
#include "problems/cosinemixture.h"
#include "problems/cassini.h"
#include "problems/ded1.h"
#include "problems/ded2.h"
#include "problems/differentpowers.h"
#include "problems/diracproblem.h"
#include "problems/eld1.h"
#include "problems/eld2.h"
#include "problems/eld3.h"
#include "problems/eld4.h"
#include "problems/eld5.h"
#include "problems/easom.h"
#include "problems/ellipsoidal.h"
#include "problems/equalmaxima.h"
#include "problems/expotential.h"
#include "problems/goldstein.h"
#include "problems/griewank.h"
#include "problems/griewankrosenbrock.h"
#include "problems/hansen.h"
#include "problems/hartmann3.h"
#include "problems/hartmann6.h"
#include "problems/rotatedrosenbrock.h"
#include "problems/shekel5.h"
#include "problems/shekel7.h"
#include "problems/shekel10.h"
#include "problems/shubert.h"
#include "problems/stepellipsoidal.h"
#include "problems/test2n.h"
#include "problems/test30n.h"
#include "problems/antennaarray.h"
#include "problems/antennaula.h"
#include "problems/bifunctionalcatalyst.h"
#include "problems/bucherastrigin.h"
#include "problems/fmsynth.h"
#include "problems/gallagher101.h"
#include "problems/gallagher21.h"
#include "problems/heatexchanger.h"
#include "problems/himmelblau.h"
#include "problems/hydrothermal.h"
#include "problems/ik6dof.h"
#include "problems/katsuura.h"
#include "problems/levy.h"
#include "problems/lunacekbirastrigin.h"
#include "problems/messenger.h"
#include "problems/michalewicz.h"
#include "problems/ofdmpower.h"
#include "problems/polyphase.h"
#include "problems/portfoliomv.h"
#include "problems/potential.h"
#include "problems/rastrigin.h"
#include "problems/rastrigin2.h"
#include "problems/rosenbrock.h"
#include "problems/schaffer.h"
#include "problems/schwefel.h"
#include "problems/sphere.h"
#include "problems/tandem.h"
#include "problems/tersoffb.h"
#include "problems/tersoffc.h"
#include "problems/tnep.h"
#include "problems/transmissionpricing.h"
#include "problems/vibratingplatform.h"
#include "problems/weierstrass.h"
#include "problems/wirelesscoverage.h"
#include "problems/zakharov.h"
#include "problems/booth.h"
#include "problems/beale.h"
#include "problems/matyas.h"
#include "problems/mccormick.h"
#include "problems/colville.h"
#include "problems/dixonprice.h"
#include "problems/trid.h"
#include "problems/powell.h"
#include "problems/alpine1.h"
#include "problems/salomon.h"
#include "problems/whitley.h"
#include "problems/perm.h"
#include "problems/cassini1.h"
#include "problems/sagas.h"
#include "problems/gtoc1.h"
#include "problems/rosetta.h"
#include "problems/stirredtankreactor.h"
#include "problems/weldedbeam.h"
#include "problems/speedreducer.h"
#include "problems/pressurevessel.h"
#include "problems/springdesign.h"
#include "problems/cantileverbeam.h"
#include "problems/threebartruss.h"
#include "problems/geartrain.h"
#include "problems/eggholder.h"
#include "problems/crossintray.h"
#include "problems/holdertable.h"
#include "problems/bukinn6.h"
#include "problems/dropwave.h"
#include "problems/langermann.h"
#include "problems/vincent.h"
#include "problems/fiveunevenpeaktrap.h"
#include "problems/gascycle.h"
#include "problems/sinusoidal.h"
#include "problems/gkls250.h"
#include "problems/gkls350.h"
#include "problems/gkls2100.h"
#include "problems/cec2022_zakharov.h"
#include "problems/cec2022_rosenbrock.h"
#include "problems/cec2022_schafferf7.h"
#include "problems/cec2022_noncontinuous_rastrigin.h"
#include "problems/cec2022_levy.h"
#include "problems/cec2022_hybrid2.h"
#include "problems/cec2022_hybrid10.h"
#include "problems/cec2022_hybrid6.h"
#include "problems/cec2022_composition1.h"
#include "problems/cec2022_composition2.h"
#include "problems/cec2022_composition6.h"
#include "problems/cec2022_composition7.h"
#include "problems/cec2017_f1.h"
#include "problems/cec2017_f3.h"
#include "problems/cec2017_f4.h"
#include "problems/cec2017_f5.h"
#include "problems/cec2017_f6.h"
#include "problems/cec2017_f7.h"
#include "problems/cec2017_f8.h"
#include "problems/cec2017_f9.h"
#include "problems/cec2017_f10.h"
#include "problems/cec2017_f11.h"
#include "problems/cec2017_f12.h"
#include "problems/cec2017_f13.h"
#include "problems/cec2017_f14.h"
#include "problems/cec2017_f15.h"
#include "problems/cec2017_f16.h"
#include "problems/cec2017_f17.h"
#include "problems/cec2017_f18.h"
#include "problems/cec2017_f19.h"
#include "problems/cec2017_f20.h"
#include "problems/cec2017_f21.h"
#include "problems/cec2017_f22.h"
#include "problems/cec2017_f23.h"
#include "problems/cec2017_f24.h"
#include "problems/cec2017_f25.h"
#include "problems/cec2017_f26.h"
#include "problems/cec2017_f27.h"
#include "problems/cec2017_f28.h"
#include "problems/cec2017_f29.h"
#include "problems/cec2017_f30.h"
#include "problems/zdt1.h"
#include "problems/zdt2.h"
#include "problems/zdt3.h"
#include "problems/zdt4.h"

#include "problems/weatherirrigation.h"
#include "problems/smartportenergy.h"
#include "problems/datacentercooling.h"// methods
#include "methods/arq.h"
#include "methods/de.h"
#include "methods/pso.h"
#include "methods/mscso.h"
#include "methods/emscso.h"
#include "methods/turbo.h"
#include "methods/rembo.h"
#include "methods/ccdg2.h"
#include "methods/cso.h"
#include "methods/lmmaes.h"
#include "methods/vkdcmaes.h"
#include "methods/rmes.h"
#include "methods/mmes.h"
#include "methods/ga.h" 
#include "methods/aco.h"
#include "methods/acor.h"
#include "methods/sa.h"
#include "methods/woa.h"
#include "methods/mewoa.h"
#include "methods/rdex.h"
#include "methods/rde.h"
#include "methods/abc.h" 
#include "methods/gwo.h" 
#include "methods/egco.h"
#include "methods/gao.h"
#include "methods/ppso.h"
#include "methods/pde.h"
#include "methods/pga.h"
#include "methods/psioa.h"
#include "methods/sioa.h"
#include "methods/sao.h"
#include "methods/psao.h"
#include "methods/bho.h"
#include "methods/tridentde.h"
#include "methods/mlshaderl.h"
#include "methods/jso.h"
#include "methods/ea4eig.h"
#include "methods/ude3.h"
#include "methods/jde.h"
#include "methods/sade.h"
#include "methods/cmaes.h"
#include "methods/clpso.h"
#include "methods/tlbo.h"
#include "methods/jaya.h"
#include "methods/sca.h"
#include "methods/fa.h"
#include "methods/ba.h"
#include "methods/hs.h"
#include "methods/cs.h"
#include "methods/so.h"
#include "methods/gsa.h"
#include "methods/alo.h"
#include "methods/hho.h"
#include "methods/mfo.h"
#include "methods/mvo.h"
#include "methods/sma.h"
#include "methods/mpa.h"
#include "methods/eo.h"
#include "methods/wca.h"
#include "methods/kh.h"
#include "methods/hba.h"
#include "methods/arq2.h"
#include "methods/ujso.h"
#include "methods/mjso.h"
#include "methods/ude.h"
#include "methods/sfcde.h"
#include "methods/awjso.h"
#include "methods/gahs.h"
#include "methods/bwjso.h"
#include "methods/bjso.h"
#include "methods/nlshadelbc.h"
#include "methods/arq3.h"
#include "methods/hades.h"
#include "methods/lracmaes.h"
#include "methods/lmcmaes.h"
#include "methods/sparq.h"


// local methods
#include "methods/gd.h"
#include "methods/nm.h"
#include "methods/lbfgs.h"
#include "methods/bfgs.h"

// multi-objective methods (optional; additive)
#include "methods/nsga2.h"
#include "methods/moead.h"
#include "methods/mopso.h"

namespace optimsolution {

std::unique_ptr<Problem> makeProblem(const std::string& raw) {
    auto name = toLower(raw);
    if (name == "rastrigin")  return std::make_unique<Rastrigin>();
    if (name == "rosenbrock") return std::make_unique<Rosenbrock>();
	if (name == "potential") return std::make_unique<Potential>();
	if (name == "ackley")    return std::make_unique<Ackley>();
	if (name == "sphere") return std::make_unique<Sphere>();
	if (name == "griewank") return std::make_unique<Griewank>();
	if (name == "levy") return std::make_unique<Levy>();
	if (name == "attractivesector") return std::make_unique<AttractiveSector>();
	if (name == "bohachevsky1") return std::make_unique<Bohachevsky1>();
	if (name == "bohachevsky2") return std::make_unique<Bohachevsky2>();
	if (name == "bohachevsky3") return std::make_unique<Bohachevsky3>();
	if (name == "branin") return std::make_unique<Branin>();
	if (name == "camel") return std::make_unique<Camel>();
	if (name == "cigar") return std::make_unique<Cigar>();
	if (name == "cosinemixture") return std::make_unique<CosineMixture>();
	if (name == "differentpowers") return std::make_unique<DifferentPowers>();
	if (name == "diracproblem") return std::make_unique<DiracProblem>();
	if (name == "easom") return std::make_unique<Easom>();
	if (name == "ellipsoidal") return std::make_unique<Ellipsoidal>();
	if (name == "equalmaxima") return std::make_unique<EqualMaxima>();
	if (name == "expotential") return std::make_unique<Expotential>();
	if (name == "goldstein") return std::make_unique<Goldstein>();
	if (name == "griewankrosenbrock") return std::make_unique<GriewankRosenbrock>();
	if (name == "hansen") return std::make_unique<Hansen>();
	if (name == "hartmann3") return std::make_unique<Hartmann3>();
	if (name == "hartmann6") return std::make_unique<Hartmann6>();
	if (name == "rastrigin2") return std::make_unique<Rastrigin2>();
	if (name == "rotatedrosenbrock") return std::make_unique<RotatedRosenbrock>();
	if (name == "shekel5")  return std::make_unique<Shekel5>();
	if (name == "shekel7")  return std::make_unique<Shekel7>();
	if (name == "shekel10") return std::make_unique<Shekel10>();
	if (name == "shubert") return std::make_unique<Shubert>();	
	if (name == "stepellipsoidal") return std::make_unique<StepEllipsoidal>();
	if (name == "test2n") return std::make_unique<optimsolution::Test2n>();
	if (name == "test30n") return std::make_unique<Test30n>();
	if (name == "antennaarray") return std::make_unique<optimsolution::AntennaArray>();
	if (name == "antennaula") {auto p = std::make_unique<optimsolution::AntennaULA>(); return p;}
	if (name == "bifunctionalcatalyst") return std::make_unique<optimsolution::BifunctionalCatalyst>();
	if (name == "bucherastrigin") return std::make_unique<optimsolution::BucheRastrigin>();
	if (name == "cassini") return std::make_unique<optimsolution::Cassini>();
	if (name == "ded1") return std::make_unique<optimsolution::DED1>();
	if (name == "ded2") return std::make_unique<optimsolution::DED2>();
	if (name == "eld1") return std::make_unique<optimsolution::ELD1>();
	if (name == "eld2") return std::make_unique<optimsolution::ELD2>();
	if (name == "eld3") return std::make_unique<optimsolution::ELD3>();
	if (name == "eld4") return std::make_unique<optimsolution::ELD4>();
	if (name == "eld5") return std::make_unique<optimsolution::ELD5>();
	if (name == "fmsynth") return std::make_unique<optimsolution::FMSynth>();
	if (name == "gallagher101") return std::make_unique<optimsolution::Gallagher101>();
	if (name == "gallagher21") return std::make_unique<optimsolution::Gallagher21>();
	if (name == "heatexchanger") return std::make_unique<optimsolution::HeatExchanger>();
	if (name == "himmelblau") return std::make_unique<optimsolution::Himmelblau>();
	if (name == "hydrothermal") return std::make_unique<optimsolution::Hydrothermal>();
	if (name == "ik6dof") return std::make_unique<optimsolution::IK6DOF>();
	if (name == "katsuura") return std::make_unique<optimsolution::Katsuura>();
	if (name == "lunacekbirastrigin") return std::make_unique<optimsolution::LunacekBiRastrigin>();
	if (name == "messenger") return std::make_unique<optimsolution::Messenger>();
	if (name == "michalewicz") return std::make_unique<optimsolution::Michalewicz>();
	if (name == "ofdmpower") return std::make_unique<optimsolution::OFDMPower>();
	if (name == "polyphase") return std::make_unique<optimsolution::Polyphase>();
	if (name == "portfoliomv") return std::make_unique<optimsolution::PortfolioMV>();
	if (name == "schaffer") return std::make_unique<optimsolution::Schaffer>();
	if (name == "schwefel") return std::make_unique<optimsolution::Schwefel>();
	if (name == "tandem") return std::make_unique<optimsolution::Tandem>();	
	if (name == "tersoffc") return std::make_unique<optimsolution::TersoffC>();
	if (name == "tersoffb") return std::make_unique<optimsolution::TersoffB>();	
	if (name == "tnep") return std::make_unique<optimsolution::TNEP>();
	if (name == "transmissionpricing") return std::make_unique<optimsolution::TransmissionPricing>();
	if (name == "vibratingplatform") return std::make_unique<optimsolution::VibratingPlatform>();
	if (name == "weierstrass") return std::make_unique<optimsolution::Weierstrass>();
	if (name == "wirelesscoverage") return std::make_unique<optimsolution::WirelessCoverage>();
	if (name == "zakharov") return std::make_unique<optimsolution::Zakharov>();
	if (name == "booth") return std::make_unique<optimsolution::Booth>();
	if (name == "beale") return std::make_unique<optimsolution::Beale>();
	if (name == "matyas") return std::make_unique<optimsolution::Matyas>();
	if (name == "mccormick") return std::make_unique<optimsolution::McCormick>();
	if (name == "colville") return std::make_unique<optimsolution::Colville>();
	if (name == "dixonprice") return std::make_unique<optimsolution::DixonPrice>();
	if (name == "trid") return std::make_unique<optimsolution::Trid>();
	if (name == "powell") return std::make_unique<optimsolution::Powell>();
	if (name == "alpine1") return std::make_unique<optimsolution::Alpine1>();
	if (name == "salomon") return std::make_unique<optimsolution::Salomon>();
	if (name == "whitley") return std::make_unique<optimsolution::Whitley>();
	if (name == "perm") return std::make_unique<optimsolution::Perm>();
	if (name == "cassini1") return std::make_unique<optimsolution::Cassini1>();
	if (name == "sagas") return std::make_unique<optimsolution::Sagas>();
	if (name == "gtoc1") return std::make_unique<optimsolution::GTOC1>();
	if (name == "rosetta") return std::make_unique<optimsolution::Rosetta>();
	if (name == "stirredtankreactor") return std::make_unique<optimsolution::StirredTankReactor>();
	if (name == "weldedbeam") return std::make_unique<optimsolution::WeldedBeam>();
	if (name == "speedreducer") return std::make_unique<optimsolution::SpeedReducer>();
	if (name == "pressurevessel") return std::make_unique<optimsolution::PressureVessel>();
	if (name == "springdesign") return std::make_unique<optimsolution::SpringDesign>();
	if (name == "cantileverbeam") return std::make_unique<optimsolution::CantileverBeam>();
	if (name == "threebartruss") return std::make_unique<optimsolution::ThreeBarTruss>();
	if (name == "geartrain") return std::make_unique<optimsolution::GearTrain>();
	if (name == "eggholder") return std::make_unique<optimsolution::Eggholder>();
	if (name == "crossintray") return std::make_unique<optimsolution::CrossInTray>();
	if (name == "holdertable") return std::make_unique<optimsolution::HolderTable>();
	if (name == "bukinn6") return std::make_unique<optimsolution::BukinN6>();
	if (name == "dropwave") return std::make_unique<optimsolution::DropWave>();
	if (name == "langermann") return std::make_unique<optimsolution::Langermann>();
	if (name == "vincent") return std::make_unique<optimsolution::Vincent>();
	if (name == "fiveunevenpeaktrap") return std::make_unique<optimsolution::FiveUnevenPeakTrap>();
	if (name == "sinusoidal") return std::make_unique<optimsolution::Sinusoidal>();
	if (name == "gascycle") return std::make_unique<optimsolution::GasCycle>();
	if (name == "gkls250") return std::make_unique<optimsolution::Gkls250>();
	if (name == "gkls350") return std::make_unique<optimsolution::Gkls350>();
	if (name == "gkls2100") return std::make_unique<optimsolution::Gkls2100>();
	if (name == "cec2022zakharov") return std::make_unique<optimsolution::CEC2022Zakharov>();	
	if (name == "cec2022rosenbrock") return std::make_unique<optimsolution::CEC2022Rosenbrock>();	
	if (name == "cec2022schafferf7") return std::make_unique<optimsolution::CEC2022SchafferF7>();	
	if (name == "cec2022noncontinuousrastrigin") return std::make_unique<optimsolution::CEC2022NoncontinuousRastrigin>();	
	if (name == "cec2022levy") return std::make_unique<optimsolution::CEC2022Levy>();	
	if (name == "cec2022hybrid2") return std::make_unique<optimsolution::CEC2022Hybrid2>();	
	if (name == "cec2022hybrid10") return std::make_unique<optimsolution::CEC2022Hybrid10>();	
	if (name == "cec2022hybrid6") return std::make_unique<optimsolution::CEC2022Hybrid6>();	
	if (name == "cec2022composition1") return std::make_unique<optimsolution::CEC2022Composition1>();	
	if (name == "cec2022composition2") return std::make_unique<optimsolution::CEC2022Composition2>();
	if (name == "cec2022composition6") return std::make_unique<optimsolution::CEC2022Composition6>();
	if (name == "cec2022composition7") return std::make_unique<optimsolution::CEC2022Composition7>();
	if (name == "cec2017f1") return std::make_unique<optimsolution::CEC2017F1>();
	if (name == "cec2017f3") return std::make_unique<optimsolution::CEC2017F3>();
	if (name == "cec2017f4") return std::make_unique<optimsolution::CEC2017F4>();
	if (name == "cec2017f5") return std::make_unique<optimsolution::CEC2017F5>();
	if (name == "cec2017f6") return std::make_unique<optimsolution::CEC2017F6>();
	if (name == "cec2017f7") return std::make_unique<optimsolution::CEC2017F7>();
	if (name == "cec2017f8") return std::make_unique<optimsolution::CEC2017F8>();
	if (name == "cec2017f9") return std::make_unique<optimsolution::CEC2017F9>();
	if (name == "cec2017f10") return std::make_unique<optimsolution::CEC2017F10>();
	if (name == "cec2017f11") return std::make_unique<optimsolution::CEC2017F11>();
	if (name == "cec2017f12") return std::make_unique<optimsolution::CEC2017F12>();
	if (name == "cec2017f13") return std::make_unique<optimsolution::CEC2017F13>();
	if (name == "cec2017f14") return std::make_unique<optimsolution::CEC2017F14>();
	if (name == "cec2017f15") return std::make_unique<optimsolution::CEC2017F15>();
	if (name == "cec2017f16") return std::make_unique<optimsolution::CEC2017F16>();
	if (name == "cec2017f17") return std::make_unique<optimsolution::CEC2017F17>();
	if (name == "cec2017f18") return std::make_unique<optimsolution::CEC2017F18>();
	if (name == "cec2017f19") return std::make_unique<optimsolution::CEC2017F19>();
	if (name == "cec2017f20") return std::make_unique<optimsolution::CEC2017F20>();
	if (name == "cec2017f21") return std::make_unique<optimsolution::CEC2017F21>();
	if (name == "cec2017f22") return std::make_unique<optimsolution::CEC2017F22>();
	if (name == "cec2017f23") return std::make_unique<optimsolution::CEC2017F23>();
	if (name == "cec2017f24") return std::make_unique<optimsolution::CEC2017F24>();
	if (name == "cec2017f25") return std::make_unique<optimsolution::CEC2017F25>();
	if (name == "cec2017f26") return std::make_unique<optimsolution::CEC2017F26>();
	if (name == "cec2017f27") return std::make_unique<optimsolution::CEC2017F27>();
	if (name == "cec2017f28") return std::make_unique<optimsolution::CEC2017F28>();
	if (name == "cec2017f29") return std::make_unique<optimsolution::CEC2017F29>();
	if (name == "cec2017f30") return std::make_unique<optimsolution::CEC2017F30>();
	if (name == "zdt1") return std::make_unique<optimsolution::ZDT1>();
	if (name == "zdt2") return std::make_unique<optimsolution::ZDT2>();
	if (name == "zdt3") return std::make_unique<optimsolution::ZDT3>();
	if (name == "zdt4") return std::make_unique<optimsolution::ZDT4>();
	
	if (name == "weatherirrigation") return std::make_unique<optimsolution::WeatherIrrigation>();	
	if (name == "smartportenergy") return std::make_unique<optimsolution::SmartPortEnergy>();	
	if (name == "datacentercooling") return std::make_unique<optimsolution::DataCenterCooling>();		
	    return nullptr;
}

std::unique_ptr<Optimizer> makeMethod(const std::string& raw) {
    auto name = toLower(raw);

    if (name == "arq")   return std::make_unique<ARQ>();
    if (name == "de")    return std::make_unique<DE>();
	if (name == "pso")   return std::unique_ptr<Optimizer>(new PSO());
	if (name == "mscso") return std::unique_ptr<Optimizer>(new MSCSO());
	if (name == "emscso") return std::unique_ptr<Optimizer>(new EMSCSO());
	if (name == "turbo") return std::unique_ptr<Optimizer>(new TuRBO());
	if (name == "rembo") return std::unique_ptr<Optimizer>(new REMBO());
	if (name == "ccdg2") return std::unique_ptr<Optimizer>(new CCDG2());
	if (name == "cso") return std::unique_ptr<Optimizer>(new CSO());
	if (name == "lmmaes") return std::unique_ptr<Optimizer>(new LMMAES());
	if (name == "vkdcmaes") return std::unique_ptr<Optimizer>(new VkDCMAES());
	if (name == "rmes") return std::unique_ptr<Optimizer>(new RmES());
	if (name == "mmes") return std::unique_ptr<Optimizer>(new MMES());
	if (name == "ga")    return std::unique_ptr<Optimizer>(new GA());
    if (name == "gd")    return std::unique_ptr<Optimizer>(new GD());
	if (name == "aco")  return std::make_unique<ACO>();	
	if (name == "acor")  return std::make_unique<ACOR>();
	if (name == "sa")    return std::make_unique<SA>();
	if (name == "woa") 	 return std::make_unique<WOA>();
	if (name == "mewoa") return std::make_unique<MEWOA>();
	if (name == "rdex")  return std::make_unique<RDEx>();
	if (name == "rde")   return std::make_unique<RDE>();
	if (name == "abc")   return std::make_unique<ABC>();
	if (name == "gwo")   return std::make_unique<GWO>();
	if (name == "egco")  return std::make_unique<EGCO>();
	if (name == "gao")   return std::make_unique<GAO>();
	if (name == "ppso") return std::make_unique<PPSO>();
	if (name == "pde") return std::make_unique<optimsolution::PDE>();
	if (name == "pga") return std::make_unique<optimsolution::PGA>();
	if (name == "psioa") return std::make_unique<optimsolution::PSIOA>();
	if (name == "sioa") return std::make_unique<optimsolution::SIOA>();
	if (name == "sao") return std::make_unique<optimsolution::SAO>();
	if (name == "psao") return std::make_unique<optimsolution::PSAO>();	
	if (name == "bho") return std::make_unique<optimsolution::BHO>();	
	if (name == "tridentde") return std::make_unique<optimsolution::TRIDENTDE>();
	if (name == "mlshaderl") return std::make_unique<optimsolution::mLSHADE_RL>();
	if (name == "jso") return std::make_unique<optimsolution::JSO>();
	if (name == "ea4eig") return std::make_unique<optimsolution::EA4Eig>();
	if (name == "ude3") return std::make_unique<optimsolution::UDE3>();	
	if (name == "jde") return std::make_unique<optimsolution::jDE>();
	if (name == "sade") return std::make_unique<optimsolution::SaDE>();
	if (name == "cmaes") return std::make_unique<optimsolution::CMAES>();
	if (name == "clpso") return std::make_unique<optimsolution::CLPSO>();	
	if (name == "tlbo") return std::make_unique<optimsolution::TLBO>();
	if (name == "jaya") return std::make_unique<optimsolution::JAYA>();
	if (name == "sca") return std::make_unique<optimsolution::SCA>();	
	if (name == "fa") return std::make_unique<optimsolution::FA>();	
	if (name == "ba") return std::make_unique<optimsolution::BA>();	
	if (name == "hs") return std::make_unique<optimsolution::HS>();	
	if (name == "cs") return std::make_unique<optimsolution::CS>();	
	if (name == "so") return std::make_unique<optimsolution::SO>();	
	if (name == "gsa") return std::make_unique<optimsolution::GSA>();
	if (name == "alo") return std::make_unique<optimsolution::ALO>();	
	if (name == "hho") return std::make_unique<optimsolution::HHO>();	
	if (name == "mfo") return std::make_unique<optimsolution::MFO>();	
	if (name == "mvo") return std::make_unique<optimsolution::MVO>();		
	if (name == "sma") return std::make_unique<optimsolution::SMA>();
	if (name == "mpa") return std::make_unique<optimsolution::MPA>();	
	if (name == "eo") return std::make_unique<optimsolution::EO>();
	if (name == "wca") return std::make_unique<optimsolution::WCA>();	
	if (name == "kh") return std::make_unique<optimsolution::KH>();
	if (name == "hba") return std::make_unique<optimsolution::HBA>();		
	if (name == "arq2") return std::make_unique<optimsolution::ARQ2>();
	if (name == "ujso") return std::make_unique<optimsolution::UJSO>();	
	if (name == "mjso") return std::make_unique<optimsolution::MJSO>();
	if (name == "ude") return std::make_unique<optimsolution::UDE>();
	if (name == "sfcde") return std::make_unique<optimsolution::SFCDE>();	
	if (name == "awjso") return std::make_unique<optimsolution::AWJSO>();	
	if (name == "gahs") return std::make_unique<optimsolution::GAHS>();
	if (name == "bwjso") return std::make_unique<optimsolution::BWJSO>();
	if (name == "bjso") return std::make_unique<optimsolution::BJSO>();	
	if (name == "nlshadelbc") return std::make_unique<optimsolution::NLSHADELBC>();	
	if (name == "arq3") return std::make_unique<optimsolution::ARQ3>();	
	if (name == "hades") return std::make_unique<optimsolution::HADES>();	
	if (name == "lracmaes") return std::make_unique<optimsolution::LRACMAES>();
	if (name == "lmcmaes") return std::make_unique<optimsolution::LMCMAES>();
	if (name == "sparq") return std::make_unique<optimsolution::SPARQ>();	
	
    if (name == "nm")    return std::unique_ptr<Optimizer>(new NM());
    if (name == "lbfgs") return std::unique_ptr<Optimizer>(new LBFGS());
    if (name == "bfgs")  return std::unique_ptr<Optimizer>(new BFGS());
	

    return nullptr;
}

std::vector<std::string> listProblemNames() {
    // NOTE: keep this list in sync with makeProblem().
    // Returned identifiers are short names accepted by the CLI/GUI.
    return {
        "datacentercooling","smartportenergy","weatherirrigation","cec2022composition7","cec2022composition6","cec2022composition2","cec2022composition1","cec2022hybrid6","cec2022hybrid10","cec2022hybrid2","cec2022levy", "cec2022noncontinuousrastrigin","cec2022schafferf7","cec2022rosenbrock","cec2022zakharov","cec2017f1","cec2017f3","cec2017f4","cec2017f5","cec2017f6","cec2017f7","cec2017f8","cec2017f9","cec2017f10","cec2017f11","cec2017f12","cec2017f13","cec2017f14","cec2017f15","cec2017f16","cec2017f17","cec2017f18","cec2017f19","cec2017f20","cec2017f21","cec2017f22","cec2017f23","cec2017f24","cec2017f25","cec2017f26","cec2017f27","cec2017f28","cec2017f29","cec2017f30","zdt1","zdt2","zdt3","zdt4","rastrigin","rosenbrock","potential","ackley","sphere","griewank","levy",
        "booth","beale","matyas","mccormick","colville","dixonprice","trid","powell","alpine1","salomon","whitley","perm",
        "cassini1","sagas","gtoc1","rosetta","stirredtankreactor",
        "weldedbeam","speedreducer","pressurevessel","springdesign","cantileverbeam","threebartruss","geartrain",
        "eggholder","crossintray","holdertable","bukinn6","dropwave","langermann","vincent","fiveunevenpeaktrap",
        "attractivesector","bohachevsky1","bohachevsky2","bohachevsky3","branin","camel",
        "cigar","cosinemixture","differentpowers","diracproblem","easom","ellipsoidal",
        "equalmaxima","expotential","goldstein","griewankrosenbrock","hansen","hartmann3",
        "hartmann6","rastrigin2","rotatedrosenbrock","shekel5","shekel7","shekel10",
        "shubert","stepellipsoidal","test2n","test30n","antennaarray","antennaula",
        "bifunctionalcatalyst","bucherastrigin","cassini","ded1","ded2","eld1","eld2",
        "eld3","eld4","eld5","fmsynth","gallagher101","gallagher21","heatexchanger",
        "himmelblau","hydrothermal","ik6dof","katsuura","lunacekbirastrigin","messenger",
        "michalewicz","ofdmpower","polyphase","portfoliomv","schaffer","schwefel",
        "tandem","tersoffc","tersoffb","tnep","transmissionpricing","vibratingplatform",
        "weierstrass","wirelesscoverage","zakharov","sinusoidal","gascycle",
        "gkls250","gkls350","gkls2100"};
}

std::vector<std::string> listMethodNames() {
    // NOTE: keep this list in sync with makeMethod().
    return {
        "sparq","mscso","emscso","turbo","rembo","ccdg2","cso","lmmaes","vkdcmaes","rmes","mmes","lmcmaes","lracmaes","hades","arq3","nlshadelbc","bjso","bwjso","gahs","awjso","sfcde","ude", "mjso","ujso","arq","arq2","de","pso","ga","gd","aco","acor","sa","woa","mewoa","abc","gwo","rdex","rde",
        "egco","gao","ppso","pde","pga","psioa","sioa","sao","psao","bho","tridentde",
        "mlshaderl","jso","ea4eig","ude3","jde","sade","cmaes","clpso","tlbo","jaya","sca","fa","ba","hs","cs","so","gsa","alo","hho","mfo","mvo","sma","mpa","eo","wca","kh","hba", "nm","lbfgs","bfgs"
    };
}

// -----------------------------------------------------------------------------
// Multi-objective (optional; additive)
// -----------------------------------------------------------------------------
bool isMultiObjectiveProblem(const std::string& name) {
    // Deliberately an explicit allowlist rather than "construct every known
    // problem and call init(2) to see what happens": several problems in
    // this codebase have effectively fixed/large dimensions baked into their
    // name and internals (e.g. gkls250, gkls350, gkls2100, ik6dof, antenna
    // arrays, TNEP's fixed 6-node/11-line network...). Probing all of them
    // with an arbitrary small dimension at GUI startup risks undefined
    // behavior (out-of-bounds access, etc.) that a try/catch cannot save,
    // rather than a safe, catchable exception. Add a name here only once a
    // problem has actually been given a numObjectives()>=2 override AND been
    // verified to init() safely at a small probe dimension.
    static const std::vector<std::string> kKnownMultiObjective = {
        "portfoliomv", "zdt1", "zdt2", "zdt3", "zdt4",
    };
    const std::string lname = toLower(name);
    for (const auto& n : kKnownMultiObjective) {
        if (n == lname) return true;
    }
    return false;
}

std::vector<std::string> listMultiObjectiveProblemNames() {
    std::vector<std::string> out;
    for (const auto& name : listProblemNames()) {
        if (isMultiObjectiveProblem(name)) out.push_back(name);
    }
    return out;
}

std::vector<std::string> listMultiObjectiveMethodNames() {
    // NOTE: keep this list in sync with makeMultiObjectiveMethod().
    return { "nsga2", "moead", "mopso" };
}

std::unique_ptr<MOOOptimizer> makeMultiObjectiveMethod(const std::string& raw) {
    auto name = toLower(raw);
    if (name == "nsga2") return std::make_unique<NSGA2>();
    if (name == "moead") return std::make_unique<MOEAD>();
    if (name == "mopso") return std::make_unique<MOPSO>();
    return nullptr;
}

} // namespace optimsolution
