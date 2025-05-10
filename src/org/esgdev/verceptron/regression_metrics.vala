public class RegressionMetrics : Object {
    public static double mse(double[] y_true, double[] y_pred) {
        double sum = 0;
        for (int i = 0; i < y_true.length; i++)
            sum += Math.pow(y_true[i] - y_pred[i], 2);
        return sum / y_true.length;
    }
    public static double mae(double[] y_true, double[] y_pred) {
        double sum = 0;
        for (int i = 0; i < y_true.length; i++)
            sum += Math.fabs(y_true[i] - y_pred[i]);
        return sum / y_true.length;
    }
    public static double max_error(double[] y_true, double[] y_pred) {
        double max_err = 0;
        for (int i = 0; i < y_true.length; i++)
            max_err = Math.fmax(max_err, Math.fabs(y_true[i] - y_pred[i]));
        return max_err;
    }
    public static double r2(double[] y_true, double[] y_pred) {
        double mean = 0;
        for (int i = 0; i < y_true.length; i++)
            mean += y_true[i];
        mean /= y_true.length;
        double ss_tot = 0, ss_res = 0;
        for (int i = 0; i < y_true.length; i++) {
            ss_tot += Math.pow(y_true[i] - mean, 2);
            ss_res += Math.pow(y_true[i] - y_pred[i], 2);
        }
        return 1.0 - (ss_res / ss_tot);
    }
}