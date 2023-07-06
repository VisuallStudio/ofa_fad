#net=mobilefadnet
net=ofa_fadnet
dataset=sceneflow

#model=models/${net}-sceneflow-v3/model_best.pth
#model=models/${net}-sceneflow.pth
model=models/${net}-sceneflow-1e-3-largeV2-kernelV2/model_best.pth
outf=detect_results/${net}-${dataset}-test/
#outf=detect_results/${net}-${dataset}/

filelist=lists/jingbo_test.list
#filelist=lists/FlyingThings3D_release_TEST.list
#filepath=/datasets
#filepath=./data
filepath=data

CUDA_VISIBLE_DEVICES=0 python detecter.py --model $model --rp $outf --filelist $filelist --filepath $filepath --devices 0 --net ${net} --format pfm
