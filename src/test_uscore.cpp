#include "uscore.h"
#include <cstdio>
#include <cassert>
#include <cmath>

using namespace optimsolution::uscore;

static bool nearlyEqual(double a, double b, double eps=1e-9) { return std::fabs(a-b) < eps; }

int main() {
    // ---------------------------------------------------------------
    // Reproduce the slides' Figure 1/2 worked example:
    //   3 algorithms (A1, A2, A3), 4 trials each, all "successful" so the
    //   ranking reduces to pure speed (hitEvals ascending = better).
    //   Expected trial order best->worst (rank 12..1):
    //     A1,A2,A1,A2,A1,A3,A2,A3,A1,A2,A3,A3
    //   Expected: SR(A1)=34,U(A1)=24 ; SR(A2)=29,U(A2)=19 ; SR(A3)=15,U(A3)=5
    // ---------------------------------------------------------------
    auto succ = [](double fe){ TrialOutcome t; t.success=true; t.hitEvals=fe; t.finalBestF=0.0; return t; };

    MethodTrials a1{"A1", { succ(1), succ(3), succ(5), succ(9) }};
    MethodTrials a2{"A2", { succ(2), succ(4), succ(7), succ(10) }};
    MethodTrials a3{"A3", { succ(6), succ(8), succ(11), succ(12) }};

    auto scores = computeUScoreForProblem({a1, a2, a3});
    assert(scores.size() == 3);
    for (auto& s : scores) {
        printf("%-3s nTrials=%d nSuccess=%d SR=%.1f U=%.1f rank=%d\n",
               s.methodName.c_str(), s.nTrials, s.nSuccess, s.rankSum, s.uScore, s.rankOnProblem);
    }
    assert(nearlyEqual(scores[0].rankSum, 34.0) && nearlyEqual(scores[0].uScore, 24.0) && scores[0].rankOnProblem==1);
    assert(nearlyEqual(scores[1].rankSum, 29.0) && nearlyEqual(scores[1].uScore, 19.0) && scores[1].rankOnProblem==2);
    assert(nearlyEqual(scores[2].rankSum, 15.0) && nearlyEqual(scores[2].uScore, 5.0)  && scores[2].rankOnProblem==3);
    printf("OK: Figure 1/2 worked example matches exactly.\n\n");

    // ---------------------------------------------------------------
    // Reproduce Table 2 of the peer-reviewed paper itself (Price, Kumar &
    // Suganthan, "Trial-based dominance...", Swarm Evol. Comput. 78,
    // 101287, 2023): algorithms P, Q, R, 4 trials each (n=4, m=3, no
    // ties). Trial order worst(rank1)->best(rank12): r,p,q,p,r,q,p,r,q,q,r,p
    // (i.e. the paper's own best(12)->worst(1) listing "p r q q r p q p r
    // q p r" reversed). Expected: SR/U = P:26/16, Q:28/18, R:24/14.
    // ---------------------------------------------------------------
    {
        // Simplest reliable approach: directly assign hitEvals = (13 - rank)
        // for each trial using the paper's own worst(1)->best(12) listing:
        //   rank:  1  2  3  4  5  6  7  8  9 10 11 12
        //   trial: r  p  q  r  p  q  r  p  q  q  r  p   (reverse of "p r q q r p q p r q p r")
        MethodTrials Pp{"P", { succ(13-2), succ(13-5), succ(13-7), succ(13-12) }}; // P at ranks 2,5,7,12
        MethodTrials Qq{"Q", { succ(13-3), succ(13-6), succ(13-9), succ(13-10) }}; // Q at ranks 3,6,9,10
        MethodTrials Rr{"R", { succ(13-1), succ(13-4), succ(13-8), succ(13-11) }}; // R at ranks 1,4,8,11

        auto sc2 = computeUScoreForProblem({Pp, Qq, Rr});
        printf("Table 2 check: P SR=%.1f U=%.1f | Q SR=%.1f U=%.1f | R SR=%.1f U=%.1f\n",
               sc2[0].rankSum, sc2[0].uScore, sc2[1].rankSum, sc2[1].uScore, sc2[2].rankSum, sc2[2].uScore);
        assert(nearlyEqual(sc2[0].rankSum, 26.0) && nearlyEqual(sc2[0].uScore, 16.0)); // P
        assert(nearlyEqual(sc2[1].rankSum, 28.0) && nearlyEqual(sc2[1].uScore, 18.0)); // Q
        assert(nearlyEqual(sc2[2].rankSum, 24.0) && nearlyEqual(sc2[2].uScore, 14.0)); // R
        printf("OK: paper's Table 2 (P,Q,R) matches exactly (26/16, 28/18, 24/14).\n\n");
    }

    // ---------------------------------------------------------------
    // Reproduce the paper's Section 3.2 tie example: m=2 algorithms, n=2
    // trials each, worst->best order "p q q p" where the two middle "q"
    // trials are EXACT ties (must share the average rank 2.5, not ranks
    // 2 and 3). Expected: U_P = U_Q = 2 (paper states "U-score for P is
    // 2+0=2 and the U-score for Q is 1+1=2").
    // ---------------------------------------------------------------
    {
        // worst->best: p(rank1), q(tied,~2.5), q(tied,~2.5), p(rank4)
        // Use identical hitEvals for the two tied "q" trials so they are
        // genuinely indistinguishable to the ranking rule.
        MethodTrials P{"P", { succ(100.0) /*worst, rank1*/, succ(1.0) /*best, rank4*/ }};
        MethodTrials Q{"Q", { succ(50.0), succ(50.0) /*exact tie -> both rank 2.5*/ }};
        auto sc3 = computeUScoreForProblem({P, Q});
        printf("Tie check: P SR=%.1f U=%.1f | Q SR=%.1f U=%.1f\n",
               sc3[0].rankSum, sc3[0].uScore, sc3[1].rankSum, sc3[1].uScore);
        assert(nearlyEqual(sc3[0].rankSum, 5.0) && nearlyEqual(sc3[0].uScore, 2.0)); // P: ranks 1+4=5, cf=3 -> U=2
        assert(nearlyEqual(sc3[1].rankSum, 5.0) && nearlyEqual(sc3[1].uScore, 2.0)); // Q: ranks 2.5+2.5=5, cf=3 -> U=2
        printf("OK: paper's Section 3.2 tie example matches exactly (U_P=U_Q=2).\n\n");
    }

    // ---------------------------------------------------------------
    // Complete-dominance edge case (as seen in the F1-F29 tables, e.g.
    // "3750/1"): with n trials per algorithm and k algorithms compared,
    // an algorithm that beats every trial of every other algorithm should
    // get U-score = n*(k-1)*n = n^2*(k-1).
    // Here: 7 algorithms, 25 trials each -> total=175, max U = 25*150=3750.
    // ---------------------------------------------------------------
    {
        std::vector<MethodTrials> methods;
        // Dominant algorithm: fastest 25 trials (hitEvals 1..25).
        std::vector<TrialOutcome> dom;
        for (int i = 1; i <= 25; ++i) dom.push_back(succ(double(i)));
        methods.push_back(MethodTrials{"DOM", dom});
        // 6 other algorithms, each with 25 trials, all slower than DOM.
        double fe = 100.0;
        for (int k = 0; k < 6; ++k) {
            std::vector<TrialOutcome> other;
            for (int i = 0; i < 25; ++i) { other.push_back(succ(fe)); fe += 1.0; }
            methods.push_back(MethodTrials{"OTHER" + std::to_string(k), other});
        }
        auto sc = computeUScoreForProblem(methods);
        printf("DOM U-score = %.1f (expected 3750)\n", sc[0].uScore);
        assert(nearlyEqual(sc[0].uScore, 3750.0));
        assert(sc[0].rankOnProblem == 1);
        printf("OK: complete-dominance edge case matches the slides' 3750 figure.\n\n");
    }

    // ---------------------------------------------------------------
    // Mixed success/failure: any success outranks any failure, and among
    // failures the lower final error wins.
    // ---------------------------------------------------------------
    {
        TrialOutcome s1; s1.success=true;  s1.hitEvals=10; s1.finalBestF=0.0;
        TrialOutcome f1; f1.success=false; f1.finalBestF=5.0;
        TrialOutcome f2; f2.success=false; f2.finalBestF=2.0; // better (lower) than f1

        MethodTrials mA{"A", {s1}};       // 1 successful trial -> should get the best rank (2)
        MethodTrials mB{"B", {f1, f2}};   // 2 failed trials -> f2 should outrank f1

        auto sc = computeUScoreForProblem({mA, mB});
        // total trials = 3; ranks: worst=1, best=3.
        // Expected order worst->best: f1 (worst), f2, s1 (best) => ranks f1=1, f2=2, s1=3
        // SR(A) = 3 ; n=1 -> cf=1 -> U(A) = 3-1=2
        // SR(B) = 1+2=3 ; n=2 -> cf=3 -> U(B) = 3-3=0
        printf("A: SR=%.1f U=%.1f | B: SR=%.1f U=%.1f\n", sc[0].rankSum, sc[0].uScore, sc[1].rankSum, sc[1].uScore);
        assert(nearlyEqual(sc[0].rankSum, 3.0) && nearlyEqual(sc[0].uScore, 2.0));
        assert(nearlyEqual(sc[1].rankSum, 3.0) && nearlyEqual(sc[1].uScore, 0.0));
        assert(sc[0].rankOnProblem == 1 && sc[1].rankOnProblem == 2);
        printf("OK: mixed success/failure ordering is correct.\n\n");
    }

    // ---------------------------------------------------------------
    // resolveTrial() / computeSharedTarget() sanity check.
    // ---------------------------------------------------------------
    {
        std::vector<double> finals = {1.0, 0.5, 2.0, 0.5};
        double target, tol;
        computeSharedTarget(finals, target, tol);
        assert(nearlyEqual(target, 0.5));
        assert(tol >= 1e-12);

        std::vector<EvalPoint> trace = { {100,10.0}, {200,1.0}, {300,0.5000000001}, {400,0.5} };
        TrialOutcome t = resolveTrial(trace, target, tol);
        assert(t.success);
        assert(nearlyEqual(t.hitEvals, 300.0));

        std::vector<EvalPoint> trace2 = { {100,10.0}, {200,3.0}, {300,2.5} };
        TrialOutcome t2 = resolveTrial(trace2, target, tol);
        assert(!t2.success);
        assert(nearlyEqual(t2.finalBestF, 2.5));
        printf("OK: resolveTrial()/computeSharedTarget() behave as expected.\n\n");
    }

    // ---------------------------------------------------------------
    // aggregateUScore(): sum across problems + overall ranking.
    // ---------------------------------------------------------------
    {
        ProblemScore p1a{"A",4,4,34,24,1}, p1b{"B",4,4,29,19,2}, p1c{"C",4,4,15,5,3};
        ProblemScore p2a{"A",4,4,20,10,2}, p2b{"B",4,4,35,25,1}, p2c{"C",4,4,23,13,3};
        auto overall = aggregateUScore({ {p1a,p1b,p1c}, {p2a,p2b,p2c} });
        for (auto& o : overall) {
            printf("%-3s nProblems=%d sumU=%.1f RS=%.1f overallRank=%d\n",
                   o.methodName.c_str(), o.nProblems, o.sumUScore, o.sumRank, o.overallRank);
        }
        // A: sumU=34, RS=1+2=3 ; B: sumU=44, RS=2+1=3 ; C: sumU=18, RS=3+3=6
        // RS ties between A and B (3 vs 3) broken by higher sumUScore -> B first.
        for (auto& o : overall) {
            if (o.methodName=="A") assert(nearlyEqual(o.sumUScore,34.0) && nearlyEqual(o.sumRank,3.0));
            if (o.methodName=="B") assert(nearlyEqual(o.sumUScore,44.0) && nearlyEqual(o.sumRank,3.0));
            if (o.methodName=="C") assert(nearlyEqual(o.sumUScore,18.0) && nearlyEqual(o.sumRank,6.0));
            if (o.methodName=="B") assert(o.overallRank==1);
            if (o.methodName=="A") assert(o.overallRank==2);
            if (o.methodName=="C") assert(o.overallRank==3);
        }
        printf("OK: aggregateUScore() sums and overall ranking are correct.\n\n");
    }

    printf("ALL TESTS PASSED.\n");
    return 0;
}
