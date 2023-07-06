# import cv2
#
# # 加载SceneFlow图像
# image_path = 'detect_results/fadnet-sceneflow/FlyingThings3D_release_frames_cleanpass_TEST_B_0020_left_0011.pfm'
# image = cv2.imread(image_path)
#
# # 显示图像
# cv2.imshow('SceneFlow Image', image)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
ks = [3, 4, 6, 3, 4, 5, 2, 1, 3, 3, 4, 1]
ds = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
es = [2, 4, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5]
cunet_ks = [1, 2, 3, 4, 5, 6, 7, 8]
cunet_ds = [4, 4, 4, 4, 4, 4, 4, 4]
cunet_es = [2, 3, 4, 5, 6, 7, 8, 9]

str = '(netconf)'
ofa_settings = {'ks': ks, 'ds': ds, 'es': es}
cuent_settings = {'cunet_ks': cunet_ks, 'cunet_ds': cunet_ds, 'cuent_es': cunet_es}
ofa_settings.update(cuent_settings)
subsettings = ofa_settings
for key, val in subsettings.items():

    str = str + ', ' + key + ':{'
    for ksval in val:
        str += '{}'.format(ksval)
    str += "}"
# print(' {}'.format(str))
# print(subsettings)
# print(len(subsettings))
dict = {}
dict_fad = {}
dict_cuentt = {}
dict['1.24'] = subsettings
dict['3.53']=subsettings
# print(dict)
str=''
for keys, values in zip(dict.keys(), dict.values()):
    count = 0
    # print(values)
    # input()
    for key, value in zip(values.keys(), values.values()):
        str=values
        if count < 3:
            dict_fad[key] = value
        else:
            dict_cuentt[key] = value
        count += 1
    print(str)
print(dict_fad)
print(dict_cuentt)
print(len(dict))
print(keys)

