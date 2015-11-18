%matplotlib inline
%load_ext autoreload
%autoreload 2

import numpy as np
import skimage.io as io

Z = np.zeros((63,376,500))

pic_array = (
'https://farm4.staticflickr.com/3913/15225473936_b0e7b83734_z_d.jpg',
'https://farm6.staticflickr.com/5588/15245381241_700ea05db1_z_d.jpg',
'https://farm4.staticflickr.com/3877/15245381211_c2dce16a2f_z_d.jpg',
'https://farm4.staticflickr.com/3921/15061717289_53c8e486d4_z_d.jpg',
'https://farm4.staticflickr.com/3890/15061918228_227a6d9b74_z_d.jpg',
'https://farm4.staticflickr.com/3889/15245380771_2eca66bea7_z_d.jpg',
'https://farm6.staticflickr.com/5565/15245380551_20e4e141c7_z_d.jpg',
'https://farm6.staticflickr.com/5570/15061909607_8af8afe624_z_d.jpg',
'https://farm6.staticflickr.com/5584/15061796840_c7a73c26d0_z_d.jpg',
'https://farm4.staticflickr.com/3921/15061909357_0a4536ed1e_z_d.jpg',
'https://farm6.staticflickr.com/5593/15061909187_ece1b3ea06_z_d.jpg',
'https://farm4.staticflickr.com/3897/15061796360_d771a63e4a_z_d.jpg',
'https://farm4.staticflickr.com/3914/15061715939_1f571a913c_z_d.jpg',
'https://farm4.staticflickr.com/3908/15225472466_58d6ca44b4_z_d.jpg',
'https://farm6.staticflickr.com/5591/15248096422_4a63a3fa55_z_d.jpg',
'https://farm4.staticflickr.com/3921/15245379711_c73333892a_z_d.jpg',
'https://farm6.staticflickr.com/5560/15061796010_c6feb5714e_z_d.jpg',
'https://farm4.staticflickr.com/3913/15245379481_fa3286bfc6_z_d.jpg',
'https://farm4.staticflickr.com/3898/15061795770_1fd51d9383_z_d.jpg',
'https://farm4.staticflickr.com/3898/15061795770_1fd51d9383_z_d.jpg',
'https://farm6.staticflickr.com/5557/15225471866_7a9b298846_z_d.jpg',
'https://farm4.staticflickr.com/3838/15248470565_56b8881cb2_z_d.jpg',
'https://farm4.staticflickr.com/3850/15245379231_919d661b53_z_d.jpg',
'https://farm4.staticflickr.com/3904/15248470295_971c077b40_z_d.jpg',
'https://farm6.staticflickr.com/5594/15245378771_1cd67c16e6_z_d.jpg',
'https://farm4.staticflickr.com/3901/15061714479_63d4e52693_z_d.jpg',
'https://farm6.staticflickr.com/5594/15248469785_471dd12a5f_z_d.jpg',
'https://farm4.staticflickr.com/3837/15061915848_46584e8279_z_d.jpg',
'https://farm4.staticflickr.com/3911/15061915828_b3e9aa67ed_z_d.jpg',
'https://farm4.staticflickr.com/3849/15061794910_289dde4d8c_z_d.jpg',
'https://farm4.staticflickr.com/3857/15248469535_d08cef2e36_z_d.jpg',
'https://farm6.staticflickr.com/5584/15061714239_e35509b83f_z_d.jpg',
'https://farm4.staticflickr.com/3835/15061907407_f0cd26759f_z_d.jpg',
'https://farm4.staticflickr.com/3872/15245378111_3a172eaab5_z_d.jpg',
'https://farm4.staticflickr.com/3899/15248094622_ccd40138ba_z_d.jpg',
'https://farm4.staticflickr.com/3869/15061794330_2c9cca24a9_z_d.jpg',
'https://farm6.staticflickr.com/5583/15061794300_ed1680901f_z_d.jpg',
'https://farm6.staticflickr.com/5583/15248469175_2039c3b0ef_z_d.jpg',
'https://farm4.staticflickr.com/3858/15248469135_51d898e620_z_d.jpg',
'https://farm4.staticflickr.com/3863/15245377731_b3bf1ca851_z_d.jpg',
'https://farm4.staticflickr.com/3835/15225470036_b3967a6f5f_z_d.jpg',
'https://farm4.staticflickr.com/3901/15061906677_b1c39a5254_z_d.jpg',
'https://farm6.staticflickr.com/5570/15061906557_3d36711e17_z_d.jpg',
'https://farm6.staticflickr.com/5582/15061906367_f4c4a586b6_z_d.jpg',
'https://farm4.staticflickr.com/3868/15061793360_1183a45351_z_d.jpg',
'https://farm4.staticflickr.com/3908/15225469326_f7202c5cb9_z_d.jpg',
'https://farm4.staticflickr.com/3925/15225469176_10f65dc211_z_d.jpg',
'https://farm4.staticflickr.com/3916/15061712399_1a2d451ce6_z_d.jpg',
'https://farm4.staticflickr.com/3915/15061905357_4907b07388_z_d.jpg',
'https://farm4.staticflickr.com/3876/15061792370_522a81fa15_z_d.jpg',
'https://farm4.staticflickr.com/3904/15245376151_c7043cb8ba_z_d.jpg',
'https://farm6.staticflickr.com/5586/15061905067_4f545e4f92_z_d.jpg',
'https://farm6.staticflickr.com/5595/15061792190_c425e4d2cc_z_d.jpg',
'https://farm4.staticflickr.com/3904/15061711949_8083bba099_z_d.jpg',
'https://farm4.staticflickr.com/3908/15248092302_3a94178821_z_d.jpg',
'https://farm6.staticflickr.com/5579/15061912968_eab32bc312_z_d.jpg',
'https://farm4.staticflickr.com/3897/15248466815_958b48c000_z_d.jpg',
'https://farm4.staticflickr.com/3897/15248466765_2354142b9a_z_d.jpg',
'https://farm4.staticflickr.com/3911/15245375521_f1e14e1ddc_z_d.jpg',
'https://farm6.staticflickr.com/5565/15248466485_78e3a05904_z_d.jpg')


for x in range(0,59)

X = io.imread(pic_array[x])[None]
X = np.round(X / float(np.max(X))).astype(int) 

Z[x] = X





