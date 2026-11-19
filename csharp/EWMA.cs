namespace EWMABiasCorrectionResearch
{
    /// <summary>
    /// Exponentially Weighted Moving Average (EWMA) with exact affine-invariant bias correction:
    ///   V_hat_t = (V_t - beta^t * V_0) / (1 - beta^t),  where t = i + 1 is the discrete step index.
    /// Fixes the original coordinate-vs-step index bug and generalizes V_0 != 0.
    /// </summary>
    public class EWMAWithBiasCorrection
    {
        private readonly Func<float, float, int, float, float> biasCorrection;
        private readonly float beta;
        private readonly float vInit;

        public EWMAWithBiasCorrection(Func<float, float, int, float, float> biasCorrection, float beta, float vInit = 0.0f)
        {
            this.biasCorrection = biasCorrection;
            this.beta = beta;
            this.vInit = vInit;
        }

        /// <summary>
        /// Backwards-compatible constructor using exact affine zeroth-order bias correction:
        ///   (Vt - beta^t * V0) / (1 - beta^t)
        /// </summary>
        public EWMAWithBiasCorrection(float beta, float vInit = 0.0f)
            : this(
                (vt, b, step, v0) =>
                {
                    float bt = (float)Math.Pow(b, step);
                    return (vt - bt * v0) / Math.Max(1.0f - bt, 1e-12f);
                },
                beta,
                vInit)
        {
        }

        public List<PointF> CalculateSeries(List<PointF> points)
        {
            List<PointF> result = new List<PointF>(points.Count);
            float vPrev = this.vInit;

            for (int i = 0; i < points.Count; i++)
            {
                int step = i + 1; // Exact discrete 1-based step index (independent of domain coordinate X)
                vPrev = (this.beta * vPrev) + ((1.0f - this.beta) * points[i].Y);
                result.Add(new PointF(points[i].X, biasCorrection(vPrev, this.beta, step, this.vInit)));
            }

            return result;
        }
    }

    public class EWMAWithoutBiasCorrection
    {
        private readonly float beta;
        private readonly float vInit;

        public EWMAWithoutBiasCorrection(float beta, float vInit = 0.0f)
        {
            this.beta = beta;
            this.vInit = vInit;
        }

        public List<PointF> CalculateSeries(List<PointF> points)
        {
            List<PointF> result = new List<PointF>(points.Count);
            float vPrev = this.vInit;

            for (int i = 0; i < points.Count; i++)
            {
                vPrev = (this.beta * vPrev) + ((1.0f - this.beta) * points[i].Y);
                result.Add(new PointF(points[i].X, vPrev));
            }

            return result;
        }
    }

    /// <summary>
    /// Enhanced Adaptive Trend- & Init-Bias-Corrected EWMA (AdaTC-EWMA) for 1D PointF series,
    /// backed by the generalized higher-dimensional operator VectorAdaTCEWMA.
    /// </summary>
    public class AdaTCEWMA
    {
        private readonly float beta;
        private readonly float vInit;

        public AdaTCEWMA(float beta, float vInit = 0.0f)
        {
            this.beta = beta;
            this.vInit = vInit;
        }

        public List<PointF> CalculateSeries(List<PointF> points)
        {
            List<PointF> result = new List<PointF>(points.Count);
            var engine = new VectorAdaTCEWMA(dimension: 1, beta: this.beta, v0: new double[] { this.vInit });
            double[] sample = new double[1];

            for (int i = 0; i < points.Count; i++)
            {
                sample[0] = points[i].Y;
                double[] estimate = engine.Step(sample);
                result.Add(new PointF(points[i].X, (float)estimate[0]));
            }

            return result;
        }
    }
}
