@echo off
REM Replan with ResidualEncoderUNet - Fixed to use local nnUNet installation

echo ======================================================================
echo Replanning Dataset001_Cervical with ResidualEncoderUNet
echo ======================================================================

REM Set environment variables
set nnUNet_raw=C:\Users\anoma\Downloads\scse-nnUNet\trial_preprocessing\nnUNet_raw
set nnUNet_preprocessed=C:\Users\anoma\Downloads\scse-nnUNet\trial_preprocessing\nnUNet_preprocessed
set nnUNet_results=C:\Users\anoma\Downloads\scse-nnUNet\trial_preprocessing\nnUNet_results

echo Environment variables:
echo nnUNet_raw: %nnUNet_raw%
echo nnUNet_preprocessed: %nnUNet_preprocessed%
echo.

REM Change to scse-nnUNet directory to use local nnUNet
cd C:\Users\anoma\Downloads\scse-nnUNet

echo Running ResidualEncoderUNet planner...
python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints plan_experiment -d 001 -pl nnUNetPlannerResEncL

echo.
echo ======================================================================
echo Done! Now running preprocessing with ResEnc plans...
echo ======================================================================

python -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints preprocess -d 001 -c 3d_fullres

echo.
echo ======================================================================
echo Complete! Dataset now uses ResidualEncoderUNet architecture.
echo ======================================================================

pause
