@echo off
REM Set nnUNet environment variables for testing with yolo_preprocessing data

echo ======================================================================
echo Setting nnUNet Environment Variables
echo ======================================================================

set nnUNet_raw=C:\Users\anoma\Downloads\scse-nnUNet\trial_preprocessing\nnUNet_raw
set nnUNet_preprocessed=C:\Users\anoma\Downloads\scse-nnUNet\trial_preprocessing\nnUNet_preprocessed
set nnUNet_results=C:\Users\anoma\Downloads\scse-nnUNet\trial_preprocessing\nnUNet_results

echo nnUNet_raw:          %nnUNet_raw%
echo nnUNet_preprocessed: %nnUNet_preprocessed%
echo nnUNet_results:      %nnUNet_results%
echo ======================================================================

REM Create results directory if it doesn't exist
if not exist "%nnUNet_results%" mkdir "%nnUNet_results%"

echo.
echo Environment ready! You can now run training commands like:
echo nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc
echo.
echo To test the environment, run:
echo nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc --npz
echo ======================================================================

REM Keep the command prompt open with variables set
cmd /k
