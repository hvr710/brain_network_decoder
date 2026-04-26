注意：
dataset4在服务器上挂载在 mnt/ 下面；
所有数据集都用的AAL_116的atlas（因为原始LCM论文里面他们全部用的AAL训练的预训练权重，所以我们现在做下游微调必须要用他们已经训练好的预训练权重，在这里这两个.pt【？？？data/】，所以我们的数据集atlas要和他们保持一致；
我们只是用LCM这个论文和repo的模型框架，但是他做的是Omni这个模型的baseline，所以下游任务要和这篇论文的table3对齐；
我给你的一些split路径可能对应的是其他的atlas文件夹底下的，这点你不用管，反正都是同一个数据集的不同atlas，split都一样


每个下游任务的微调逻辑：【！！！重要！！！】
1、用上预训练权重D:\NCClab\LCM_omni_table3_10ds\pretrain_weights_fold0/2、不要按照LCM那个论文里的多个fold（也就是多种split方式的逻辑），就按照固定每个数据集的split（所以你这里也不要再用原生repo里面的split实现了，直接参照我们对每个数据集已经分好的split，我在下面每个数据集下的split字段标注好了）；
每个数据集的每个下游任务就只用这一个split，然后设置随机种子跑多次run，每个run多个epoch，最后的评价指标也按照这个逻辑重新设计，不要再用原始repo和LCM论文里面那套评价指标了，指标设计最好和table3对齐。不然又要出现NAN了。【特例：NKI、SALD、BHRC这三个数据集因为已经有了三套已经分好的split，所以走3-fold路线？我不知道在深度学习基座模型这个领域一般怎么设计这种，是要同时多个fold多个run多个epoch还是怎么样，而且如果有三个fold的话但是我现在只有一个预训练权重（来自LCM那篇文章），你自己根据Omni论文以及你自己的判断来设计。适不适合走3-fold，还是从那三套里面任选一套跑】

label：  dataset1/ningzh/labels
split:  原始dataset4/


LCM下游任务:

一、年龄回归  （对应table3：Age Regression）（指标：MSE 、R）

1、ABIDE：871 //单sub  ❌️（609？）
  原始时间序列（.npy）： 
    "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABIDE\AAL"
  label标签数据（.csv）：
    "\\10.16.57.94\dataset1\ningzh\labels\age\ABIDE.csv"
  split（.txt）：
    "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABIDE\Schaefer2018_100_crop_split\val.txt"
    "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABIDE\Schaefer2018_100_crop_split\test.txt"
    "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABIDE\Schaefer2018_100_crop_split\train.txt"
  【这里有一个支线任务：上面这三个split文件是crop之后的，相当于每个.txt里面每个被试编号对应了两个文件，一个seg0一个seg1，你把每个.txt里面有哪些被试编号提取出来就好，这些和crop之前是一样的，提取完之后把train/test/val这三组分别有多少个被试给我统计出来，并且检查一下加起来是不是871】


2、NKI：717 //单sub  ❌️（331？？？）
   原始时间序列（.npy）：
     \\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\AAL

   label标签数据（.csv）：
   "\\10.16.57.94\dataset1\ningzh\labels\age\NKI.csv"  

   split（.txt）：⭐
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\train1.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\val1.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\test1.txt"



3、SALD：493 //单sub  ✅️（493）
    原始时间序列（.npy）：
      \\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\AAL

    label标签数据（.csv）：
   "\\10.16.57.94\dataset1\ningzh\labels\age\SALD.csv"

    split（.txt）：⭐
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\train1.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\val1.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\test1.txt"

     

二、性别分类（对应table3：Sex Classif.） （指标：ACC 、F1）

1、ABCD：1000 //crop2=run1+run2   ❌️（1680sub？？？）
   原始时间序列（.npy）【每个sub=run1/run2？？？】+ split（.npy已经按照train/va/test 分别放到了这三个文件夹）：
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABCD\AAL\train"     #train:699
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABCD\AAL\val"       #val:100
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABCD\AAL\test"      #test:201

   label标签数据（.csv）：
"\\10.16.57.94\dataset1\ningzh\labels\sex\ABCD.csv"


2、HCP：       //6 crop     ❌️（606？？？）
   原始时间序列（.npy）【每个sub=6window ？？？】+ split（.npy已经按照train/va/test 分别放到了这三个文件夹）：
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\HCP\AAL\train"     #train:3635
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\HCP\AAL\test"      #test:1211
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\HCP\AAL\val"       #val:1211
   label标签数据（.csv）：
"\\10.16.57.94\dataset1\ningzh\labels\sex\HCP.csv"

3、BHRC：465 //单sub  ✅️（465）
    原始时间序列（.npy）：
\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\AAL

    label标签数据（.csv）：
"\\10.16.57.94\dataset1\ningzh\labels\sex\BHRC.csv"

    split（.txt）：
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\test1.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\train1.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\val1.txt"


三、疾病分类（对应table3：Diagnosis）
1、PPMI：474 //单sub    ❌️（331？？？）
   原始时间序列（.npy）：
\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\PPMI\AAL
   label标签数据（.csv）：
"\\10.16.57.94\dataset1\ningzh\labels\disease\PPMI.csv"
   split:参考另一个atlas底下的.npy的文件夹分类方式
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\PPMI\100ROI\val"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\PPMI\100ROI\test"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\PPMI\100ROI\train"
【把这三个文件夹里面对应的被试编号取出来，对应他们也就是咱们PPMI的split，记得看三个加一起个数和474是否对应的上】


2、ADNI（MCI）：（CN和MCI二分类）382//单sub  ✅️（三组和 497）
   原始时间序列（.npy）：⭐已更新
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\MCI"           #MCI：146
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\CN"             #CN：236

   label标签数据（.csv）：
"\\10.16.57.94\dataset1\ningzh\labels\disease\adni_list.xlsx"

   split（.txt）：⭐已更新
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\MCI_val_abs_AAL.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\MCI_test_abs_AAL.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\MCI_train_abs_AAL.txt"


3、ADNI（AD）：（CN和AD二分类） 351//单sub   ✅️（三组和 497）
   原始时间序列（.npy）：⭐已更新
 "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\AD"     #AD：42
 "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\CN"       #CN：236

   label标签数据（.csv）：
"\\10.16.57.94\dataset1\ningzh\labels\disease\adni_list.xlsx"

   split（.txt）：⭐已更新
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\AD_val_abs_AAL.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\AD_test_abs_AAL.txt"
"\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI\AAL\AD_train_abs_AAL.txt"


四、教育水平分类（对应table3：Education Classif.）
   NKI，同上
