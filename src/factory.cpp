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
#include "problems/gascycle.h"
#include "problems/sinusoidal.h"

#include "problems/gkls250.h"
#include "problems/gkls350.h"
#include "problems/gkls2100.h"

// methods
#include "methods/arq.h"
#include "methods/de.h"
#include "methods/pso.h"
#include "methods/ga.h"
#include "methods/aco.h"  
#include "methods/acor.h"
#include "methods/sa.h"
#include "methods/woa.h"
#include "methods/mewoa.h"
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
#include "methods/aarq.h"
#include "methods/arqeig.h"
#include "methods/fuse.h"
#include "methods/arqeigrl.h"
#include "methods/gderl.h"
#include "methods/garq.h"
#include "methods/polyde.h"
#include "methods/rarq.h"
#include "methods/gde.h"
#include "methods/arqdp.h"
#include "methods/hjso.h"
#include "methods/hde.h"
#include "methods/clpso.h"

// local methods
#include "methods/gd.h"
#include "methods/nm.h"
#include "methods/lbfgs.h"
#include "methods/bfgs.h"

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
	if (name == "sinusoidal") return std::make_unique<optimsolution::Sinusoidal>();
	if (name == "gascycle") return std::make_unique<optimsolution::GasCycle>();
	if (name == "gkls250") return std::make_unique<optimsolution::GasCycle>();
	if (name == "gkls350") return std::make_unique<optimsolution::GasCycle>();
	if (name == "gkls2100") return std::make_unique<optimsolution::GasCycle>();
    return nullptr;
}

std::unique_ptr<Optimizer> makeMethod(const std::string& raw) {
    auto name = toLower(raw);

    if (name == "arq")   return std::make_unique<ARQ>();
    if (name == "de")    return std::make_unique<DE>();
	if (name == "pso")   return std::unique_ptr<Optimizer>(new PSO());
	if (name == "ga")    return std::unique_ptr<Optimizer>(new GA());
    if (name == "gd")    return std::unique_ptr<Optimizer>(new GD());
	if (name == "aco")   return std::make_unique<ACO>();
	if (name == "acor")  return std::make_unique<ACOR>();
	if (name == "sa")    return std::make_unique<SA>();
	if (name == "woa") 	 return std::make_unique<WOA>();
	if (name == "mewoa") return std::make_unique<MEWOA>();
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
	if (name == "aarq") return std::make_unique<optimsolution::AARQ>();
	if (name == "arqeig") return std::make_unique<optimsolution::ARQEig>();
	if (name == "fuse") return std::make_unique<FUSE>();
	if (name == "arqeigrl") return std::make_unique<ARQEigRL>();
	if (name == "gderl") return std::make_unique<GDERL>();
	if (name == "garq") return std::make_unique<GARQ>();
	if (name == "polyde") return std::make_unique<PolyphaseDE>();
	if (name == "rarq") return std::make_unique<RARQ>();
	if (name == "gde") return std::make_unique<GDE>();
	if (name == "arqdp") return std::make_unique<optimsolution::ARQDP>();
	if (name == "hjso") return std::make_unique<optimsolution::HJSO>();
	if (name == "hde") return std::make_unique<optimsolution::HDE>();
	if (name == "clpso") return std::make_unique<optimsolution::CLPSO>();	
	
	

	
    if (name == "nm")    return std::unique_ptr<Optimizer>(new NM());
    if (name == "lbfgs") return std::unique_ptr<Optimizer>(new LBFGS());
    if (name == "bfgs")  return std::unique_ptr<Optimizer>(new BFGS());
	

    return nullptr;
}

} // namespace optimsolution
