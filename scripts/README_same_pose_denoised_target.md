# Same-Pose Denoised Target for Self-Detection MLP

## 1. Purpose

The current MLP learns:

    raw(θ) = f(θ) + ε

where: - θ = joint angles - f(θ) = self-influence baseline - ε = sensor
noise + small drift

Because raw fluctuates even at the same pose, the model unintentionally
learns noise. This limits validation performance.

This document describes a denoising strategy to improve prediction
performance without changing the joint-only MLP structure.

------------------------------------------------------------------------

## 2. Core Idea

For each identical joint pose, compute a representative raw value (mean
or median), and train the MLP using that denoised target.

Instead of learning noisy raw values, the model learns:

    f(θ) ≈ E[raw | θ]

This removes stochastic sensor noise from the training target.

------------------------------------------------------------------------

## 3. Algorithm

### Step 1 --- Pose Binning

Quantize joint angles:

    quant_deg = 0.05
    pose_bin = round(joint_deg / quant_deg)

Recommended resolution: - 0.05° \~ 0.1°

------------------------------------------------------------------------

### Step 2 --- Group by Same Pose

Collect raw values belonging to identical pose bins.

------------------------------------------------------------------------

### Step 3 --- Compute Denoised Target

Option A (Mean):

    y_pose = mean(raw_values)

Option B (Recommended: Median):

    y_pose = median(raw_values)

Median is more robust when sensor instability exists.

------------------------------------------------------------------------

### Step 4 --- Replace Training Target

Original: Y_train = raw

Modified: Y_train = pose_mean_raw

Now the model learns only the deterministic self-signal.

------------------------------------------------------------------------

## 4. Mathematical Interpretation

Sensor model:

    raw = f(θ) + ε
    E[ε] = 0

Taking expectation over same pose:

    E[raw | θ] = f(θ)

Therefore, pose-mean directly estimates the desired self baseline.

------------------------------------------------------------------------

## 5. Expected Improvements

  Item                   Original           Denoised
  ---------------------- ------------------ -------------
  Target variance        High               Reduced
  Noise memorization     Possible           Reduced
  Validation stability   Fluctuating        More stable
  Performance floor      Limited by noise   Lowered

------------------------------------------------------------------------

## 6. Recommended Settings

-   Pose bin resolution: 0.05° \~ 0.1°
-   Statistic: Median (if unstable sensor)
-   Minimum samples per bin: ≥ 3
-   Combine with online bias compensation for drift handling

------------------------------------------------------------------------

## 7. Drift Compensation (Recommended)

Inference equation:

    r(t) = raw(t) - f_hat(θ(t)) - b(t)

Online bias update (only in self-only region):

    b(t+1) = (1 - α) b(t) + α r(t)

This separates slow drift from pose-based self compensation.

------------------------------------------------------------------------

## 8. When To Use

Use this method if:

-   Same pose shows raw fluctuation
-   Validation STD stops decreasing
-   Sensor has instability
-   Joint-only MLP is already implemented

------------------------------------------------------------------------

## 9. Conclusion

Same-Pose Denoised Target improves joint-only residual MLP by removing
noise from supervision rather than increasing network depth.

This method is especially effective for raw3/raw6 channels where
self-influence dominates and sensor instability exists.
