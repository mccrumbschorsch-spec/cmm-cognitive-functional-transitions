
clear all
set more off
set maxvar 120000
do "stata_paths.do"
global root "$mhas_root"
***************************** Note: set the authorised cohort root in stata_paths.do before running ***************************
global dofiles=      "$root/Dofiles"         
global raw_data=     "$root/Raw_data"
global working_data= "$root/Working_data"
global temp_data=    "$root/Temp_data"

cap mkdir "$raw_data"      // 自动创建文件夹
cap mkdir "$temp_data"     // `cap` 命令可让错误的代码继续运行
cap mkdir "$working_data"    
cap mkdir "$dofiles"       // 如果已经创建了这些文件夹，也可以运行


********************************************************************************
use "$raw_data/wave5/Core survey Data/sect_a_c_d_f_e_pc_h_i_2018.dta",clear
merge m:1 cunicah subhog_18 using  "$raw_data/wave5/Core survey Data/sect_g_j_k_sa_2018.dta",nogen
merge 1:1 cunicah np using  "$raw_data/wave5/Hair Samples/MHAS 2018 - Heavy Metals Exposure.dta",nogen
keep cunicah np
save "$temp_data/wave5.dta",replace