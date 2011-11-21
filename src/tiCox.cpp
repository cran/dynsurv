#define R_PACKAGE

#include "TimeIndepCoxModel.h"
#include "TimeVaryingCoxModel.h"
#include "DynamicCoxModel.h"
#include "GibbsSampler.h"

using namespace ir;

extern "C" {
	/* Time independent coef Cox model */
	void tiCox(double *p_LRX, int *p_N, int *p_nBeta,
			   double *p_grid, int *p_K,
			   char **p_out,
			   double *p_shape, double *p_rate, double *p_mean, double *p_sd,
			   int *p_iter, int *p_burn, int *p_thin, int *p_verbose, int *p_nReport)
	{
		const Size N = p_N[0];
		const Size nBeta = p_nBeta[0];
		const Size K = p_K[0];

		ublas::matrix<double> LRX(N, nBeta + 2, 0.0);
		for (Size i = 0; i < LRX.size1(); ++i)
			for (Size j = 0; j < LRX.size2(); ++j)
				LRX(i, j) = p_LRX[i + N * j];

		ublas::vector<double> grid(K, 1.0);
		for (Size k = 0; k < grid.size(); ++k)
			grid(k) = p_grid[k];
	
		/* Pointer to data */
		boost::shared_ptr<IntRegData> pd(new IntRegData(LRX, grid));

		typedef CoxPrior<GammaPrior, NormalPrior> GammaNormal;
		GammaNormal prior(GammaPrior(p_shape[0], p_rate[0]), NormalPrior(p_mean[0], p_sd[0]));
		typedef TimeIndepCoxModel<GammaNormal > TimeIndepCox;
		boost::shared_ptr<TimeIndepCox> pm(new TimeIndepCox(pd));

		GibbsSampler<TimeIndepCox> gs(pm, p_iter[0]);

		GetRNGstate();
		gs.runGibbs(prior, static_cast<bool>(p_verbose[0]), p_nReport[0]);
		PutRNGstate();

		gs.summaryFit(std::cout, p_burn[0], p_thin[0]);
	
		std::string ss(p_out[0]);
		std::ofstream os(ss.c_str());
		gs.outputSample(os);
	}
}
