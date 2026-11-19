namespace EWMABiasCorrectionResearch
{
    /// <summary>
    /// Generalized Higher-Dimensional Adaptive Trend- & Init-Corrected Operator EWMA (AdaTC-EWMA) in R^d.
    /// Implements Theorems 1, 2, and 3 in O(d) time and O(d) memory per step:
    ///   - Arbitrary initialization compensation: v0, u0 in R^d
    ///   - Coordinate-wise anisotropic decay operator: B = diag(beta_1, ..., beta_d)
    ///   - Exact finite-t 1st-order trend lag elimination via K_t(B)
    ///   - Blind trend-invariant 2nd-difference noise estimation & optimal Wiener/James-Stein shrinkage gate gamma_hat_t.
    /// </summary>
    public sealed class VectorAdaTCEWMA
    {
        private readonly int d;
        private readonly double[] beta;
        private readonly double[] v0;
        private readonly double[] u0;
        private readonly double[] v;
        private readonly double[] u;
        private readonly double[] sigma2Raw;
        private readonly double noiseBeta;

        private double[]? xPrev1;
        private double[]? xPrev2;
        private int t;
        private int noiseSteps;

        public double LastShrinkageGate { get; private set; }

        public VectorAdaTCEWMA(int dimension, double beta = 0.9, double[]? v0 = null, double[]? u0 = null, double noiseBeta = 0.95)
            : this(CreateUniformArray(dimension, beta), v0, u0, noiseBeta)
        {
        }

        public VectorAdaTCEWMA(double[] anisotropicBeta, double[]? v0 = null, double[]? u0 = null, double noiseBeta = 0.95)
        {
            if (anisotropicBeta == null || anisotropicBeta.Length == 0)
                throw new ArgumentException("Decay vector must be non-empty.", nameof(anisotropicBeta));

            this.d = anisotropicBeta.Length;
            this.beta = (double[])anisotropicBeta.Clone();
            this.v0 = v0 != null ? (double[])v0.Clone() : new double[d];
            this.u0 = u0 != null ? (double[])u0.Clone() : (double[])this.v0.Clone();
            this.v = (double[])this.v0.Clone();
            this.u = (double[])this.u0.Clone();
            this.sigma2Raw = new double[d];
            this.noiseBeta = noiseBeta;
        }

        public double[] Step(ReadOnlySpan<double> x)
        {
            if (x.Length != d)
                throw new ArgumentException($"Expected vector of dimension {d}, got {x.Length}.");

            t++;
            double[] sigma2Hat = new double[d];

            if (xPrev1 != null && xPrev2 != null)
            {
                noiseSteps++;
                double normFactor = 1.0 - Math.Pow(noiseBeta, noiseSteps);
                for (int i = 0; i < d; i++)
                {
                    double d2 = x[i] - 2.0 * xPrev1[i] + xPrev2[i];
                    double instVar = (d2 * d2) / 6.0;
                    sigma2Raw[i] = noiseBeta * sigma2Raw[i] + (1.0 - noiseBeta) * instVar;
                    sigma2Hat[i] = sigma2Raw[i] / Math.Max(normFactor, 1e-15);
                }
            }
            else if (xPrev1 != null)
            {
                for (int i = 0; i < d; i++)
                {
                    double d1 = x[i] - xPrev1[i];
                    sigma2Hat[i] = (d1 * d1) / 2.0;
                }
            }

            xPrev2 = xPrev1;
            xPrev1 = x.ToArray();

            double[] vHat0 = new double[d];
            double[] delta = new double[d];
            double deltaNorm2 = 0.0;
            double noiseTraceTerm = 0.0;

            for (int i = 0; i < d; i++)
            {
                double b = beta[i];
                double oneMinusB = 1.0 - b;
                v[i] = b * v[i] + oneMinusB * x[i];
                u[i] = b * u[i] + oneMinusB * v[i];

                double bt = Math.Pow(b, t);
                double oneMinusBt = Math.Max(1.0 - bt, 1e-15);
                vHat0[i] = (v[i] - bt * v0[i]) / oneMinusBt;

                if (t >= 2)
                {
                    double Wt0 = Math.Max(1.0 - bt - t * bt * oneMinusB, 1e-15);
                    double uHat0 = (u[i] - bt * u0[i] - t * bt * oneMinusB * v0[i]) / Wt0;

                    double Lt = (b / oneMinusB) - (t * bt / oneMinusBt);
                    double MtNum = 2.0 * (b / oneMinusB) * oneMinusBt - 2.0 * t * bt - t * (t - 1.0) * bt * oneMinusB;
                    double Mt = MtNum / Wt0;
                    double Kt = Lt / Math.Max(Mt - Lt, 1e-15);

                    delta[i] = Kt * (vHat0[i] - uHat0);
                    deltaNorm2 += delta[i] * delta[i];

                    // Closed-form O(1) polylog-geometric variance multipliers
                    double r = b * b;
                    double rt = bt * bt;
                    double oneMinusR = Math.Max(1.0 - r, 1e-15);
                    double S0 = (1.0 - rt) / oneMinusR;
                    double S1 = (1.0 - (t + 1.0) * rt + t * rt * r) / (oneMinusR * oneMinusR);
                    double S2 = (1.0 + r - ((t + 1.0) * (t + 1.0)) * rt
                        + (2.0 * t * t + 2.0 * t - 1.0) * rt * r
                        - (t * t) * rt * r * r) / (oneMinusR * oneMinusR * oneMinusR);

                    double sumA2 = Math.Pow(oneMinusB / oneMinusBt, 2) * S0;
                    double sumAC = (Math.Pow(oneMinusB, 3) / (oneMinusBt * Wt0)) * S1;
                    double sumC2 = (Math.Pow(oneMinusB, 4) / (Wt0 * Wt0)) * S2;

                    double qDelta = (Kt * Kt) * Math.Max(sumA2 - 2.0 * sumAC + sumC2, 0.0);
                    double qVDelta = Kt * (sumA2 - sumAC);
                    noiseTraceTerm += sigma2Hat[i] * (qDelta + qVDelta);
                }
            }

            if (t == 1)
            {
                LastShrinkageGate = 0.0;
                return vHat0;
            }

            double gammaHat;
            if (t == 2)
            {
                gammaHat = 0.5;
            }
            else if (deltaNorm2 <= 1e-18)
            {
                gammaHat = 0.0;
            }
            else
            {
                gammaHat = Math.Clamp(1.0 - noiseTraceTerm / deltaNorm2, 0.0, 1.0);
            }

            LastShrinkageGate = gammaHat;
            double[] result = new double[d];
            for (int i = 0; i < d; i++)
                result[i] = vHat0[i] + gammaHat * delta[i];

            return result;
        }

        private static double[] CreateUniformArray(int length, double val)
        {
            double[] arr = new double[length];
            Array.Fill(arr, val);
            return arr;
        }
    }
}
