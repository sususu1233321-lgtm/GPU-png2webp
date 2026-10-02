"""GPU closed-loop encoder kernel (one CUDA thread per image).

The closed loop is sequential per image by algorithm (each macroblock's
prediction uses the reconstructed pixels of its left/above neighbours), so
the wavefront cannot be parallelised inside one image.  Instead we exploit
batch-level parallelism: one thread runs the ENTIRE sequential loop for one
image; a batch of 32-64 images runs as 32-64 threads in a single kernel
launch.  All arithmetic is a 1:1 port of closed_loop_jit.py (int64
fixed-point) so the emitted levels are bit-identical to the CPU version.
"""
import numpy as np
import cupy as cp

import base64, zlib as _zlib
from . import _prot
_CUDA_SRC = _prot.dec(
    "KZ=NCj3%yYpwGu3p5~MWoMI7nHBcRb9;|xk=I%k)f`g$k*2F0NsG)l2YH*eG=!e_W%?geei`uh<VJxL!w-f>~mu(r96)qb(Q"
    "v7#4NZEpl!iTN^f%-iUykjP}e!<QGuo9%0|AzkogL}UqpozrKKZh}(mM#)i+goXVL9f^^CGD=9v!Ofg-ER;~xXK=EmoYgyB+"
    "(L&obFr&C#yj&Zr4c;Pkr5C@0g(a=EHy(*@zm(bmqE(?>A09=?PDp4xcmfkFLVFWql3Y8duWXHw<~4!=}x_mKU0m0n9XW{=="
    "~SgQwTyfX>|Ja!R#Xm{G^t<B;gE9z^Zi_`FQ7AOx^Bw9SZq6}h120}+jGTxF~1Q=_KONTnS=8&!c{xVwHdLdjAcoW_hxZ@gO"
    "sMot2p2aF#BjCF?RKVO9+1?Zbh9R^yo^;np*fbyQ^+y|kDLIiQi3SgkSMGgTvQBjkAl-+YsL3yOY&OGBmVo+7E_xGXI)mLXB"
    "`pHwrr8jyTe5K7}8-WGG$+|nelK|%1-T`#!l9q+bo`xr=xXz-$)ZrEF8OfZ0WdeVi_>U)fFRk)Vp<(liWX31YSGg6#j<Hz<r"
    "isjB&yi~qR6oX|gBN^Gdk*$T`=(AMgvObll+J<>Pfc^1j+*Y}Pa|q*YgjBIhDZU&Z$ap`Kw*W<7QPym8buImT7o-3hoXB+|G"
    "Pc~rCmS+T6zN#LJ+K6KNkz9h(+wVF(w9rSs7A~Ua@SjArbE{9?0W9edyXW`$$K2#24`He3{S!eWmeIRng%3#-#IhnvZmyH!Y"
    "bfG`LVp@aq*y0-_qz3kP6Ra}X_`WPd1c^#kuRW@@=fQ5M+tQ(IyWM@V}hTd}5&2Ku72a2{)T$h5hBP&9A12TWr~f{W#2Ns|5"
    "CiE7{&9=&5XdRaeDqW_Ji{`fxFpKtnF(~(8Na$1Q&5-8rzck^xuP;UbF-21d2{3Pr9!}#h&2q6!bBequ5Gyh=S0k-@}VDQQq"
    "y>9=d_I9UifsPstQZu2+*EdOk@3v>R$PYi*Y_y>E16Lxy;90*QJ&nPh{2p`N=AE3*DtHI1?RfgIhihQ0y-;)0oLK-qbz6s}m"
    "q-7@YWP3tL(-N<!JG{?3cdr*Yyxexik%)`*gl&kCN+%7JL8lB1_i$S!(P&!o;i2n1${Sdq}oC{rqENi9P;_1M=t7bVX$N~db"
    "QThH(o|@^ZocT03gh$3X%6gCTBbJ-Al_6GORqJGMYaYUr;L9Wq!781=LtZ-*X~yD;trHSeHwSR_tjbd7kA+z7Y=c($FFPOuv"
    "%rHu=;oD{YWtv`O1sh4BtrF=#9j7GpB<UM8~!zWPLXf92C=Nuc=<qt#p&s$&~B@(wWa?n$G6_DMVPz;~l>tOeRXn=`@$>Ku%"
    "d0SyWIo^1X(_QmGNpO{g{97&iWtwIWWIaD~afPvmKaWZ`=Mm^^{kS6<yXHg5$CFo3+&Cv}b2CSsqJm!x5OxqLZ+hiOvw<XJp"
    "Bp-}$)>~iuXSm7-42rx|lh3UxsZu-J%R~OuF*h3CuwS;Rd!|Z>CLYW9m?F3C^Q_osjx6%hak7I{3W6_ej2dRiUI7T%L6(IQ^"
    "K5*YA04xtOZgG%%Hezf>@?OA|HqR6rnwzj8|B86`*9<m%CUoICwl$QH=L4LWPCW~eNfHaqO(dQxOp`gjYBV%g_Xo^@#~_yZi"
    "@MX1hoO_E6|;0aQgN7S=v}97zI|dZ+xfW14s*HkW3F&g7RQ<j+C5IUxSD+A;6r{BC9@x3q1o1se(B1d>j@LP4Fd-`D(&5u}J"
    "M$q;kMvkbS0<f!>FfyU4R!SMiVr<n3a(0x0v|0sevuYOEl%UX~VI*J2?QT4Qy3<bUP}acS^Ip5o<X7bPIQ$O-Re`hDYIK25u"
    "OD@=ytcyR!EV025ITh!+ZG{U|7hdpxf-;rY-g+Fa7&c@vCiEG3&b2rTmi40*OwsDSnjI-rhW93FVh4fzVa$kfk36=f$@0pQr"
    "H3mJT+mq}{V;(``;IClt!~XEaL?6`Yl=bp(5UmiHSLJ%+fEd9{Jgh#I<j;U(W2VRaN11lPBAlwrT1KgAj3bcaN1uqd9c|!e`"
    "CX+Amx`nCK<2-YcO9~F^n~Y{#v6V26G1?NvqM|d+0<@t_Llcik_K#g-xZAw7S7er<R8?WQYJ7$0x%9P>xj(~n-nbYWR$<{nX"
    "mPY!apv<g=%Y`2Q*}!0yg<e*&AaE_OKCIp=G@?b6%(PMWnuMH=SzkEJOF8%gs8rk_nSbQ4+^%5Y=kFnWt}3T%kYFt&~fe%=I"
    "xrDDm1#8V(mqmxA~4k6Jns!6d<$ct-;`d~E?Ra$09^Ov&zIYlSfCFzR&xtiNT&y~oe%j=ie73+q`n>$rr-?b)X21>gEpU|bU"
    "liUT{<uke=8N_IiiAD#Ra+A4E*Ndvb&=6uzU<3aV#dd|LdolfGp=uf!J&NNamCap7fVh~qh>3ys02k<^U&uj6rxPa!C<|H~+"
    ">^*Ez!jq0Bz>kYjN{nfTE@wNa1fn3?fyG7ly`&WNazXtn$sv$~w}b)<p!7pm-p045*8n@=e~oeP)^ojzdEW0y<^2cn0?Mbp%"
    "}>&mvO6C-l%CKy2}(?HuPkLv;^7nBuL11Q{&Rh%vs%B;o~2jsvy?mQG%#NtZLTnO>6!)yjXO9&hL4^)JWl4VC_ZwJ3s=+o6="
    "aIghdG+puVg-$D>KeeBDeI$%0a+wbHR^x#h9>cJB>O6WQX8#72Tw5N4b3=>M6UCK^>$a#=S{=T#Guhx~7#X%;&cL0&V9`zQH"
    "dUDXg-roT@co4ehEr-{*r+d_}h`qqJ>7I-g`khI@vIGhNMKLmVePj>)}rBbpkuu#380{8kqrnD4g+rZNidjh_FCc7K9>zD~("
    "n2N!~OJ{d>pu{z5Aum4NcY^u#r{q444>Ig+|8n}i{i4pbqxT+o8R<d2ePnA#=tqSV{C16JOW~kv5p&>tuk&7%-|KI)2kojg{"
    "r`-(0u&SHwX;yxHcR9o-LO(-LBKt~+4sLe5eM!*|@XvZ@8$KHA+I`9Fsq;+xxV8L})VH$m_wL)T4(teiA;Z68BIgm?1@K3y3"
    "F^O2K3rSw$=9q;$Agtn*zw6Ma?W^f?uJH`Z}!Zw@Xf>$>!eqC>Y_Ew^{my77$u~4;C*#k3=^#Ew_eWSB(^7}!X^ix{%SxYrn"
    "5zLwy<@FpfTu9+1zB^8jvQLSXup#<S2ivTacO%;>)TK<;jb#pwyYA1rFaE8VT*p64FkeI7*zYX1AKD;cFEZ>K-X=Cx$d-syc"
    "^&%`Ab<WnRolxxur2IRLoBBcw)1bJ>Y+8d8SMasa`UL}WpTrAS!>ER6RWWaA>vH2lOvX7}U&bqhU8@Di!y_eU38BD~6m5v?m"
    "0c;Yx5dRHrl=~i6Fk@+Z+FkaoHxf(B8C8#SbzDfMqR^IDioD)PFV1wNLnFRAo@s6qZL=5m3m2{f*C{pyX2hH8Pkei|yW-)u!"
    "er{x*4`5Vh-kE}lG+~!~nNo@D<N&9Q;PGXv&Uc`I9XYX?;~Xo#(ZajqW`Up&=Q}!S(;YrZr%n_sX>G+<iHc~;2{tCLvWVF;K"
    "^m@YZuN3Nw!9BtWNb-(>nkql84JV%BOX^5R`*s3!^BAG5mtS<QCZYC2Y;^cIdiq}KPI%RqFe&^NV0ujfdBTDig;7$H-QoG+)"
    "P^31I;W=%|LN8Gl3S@%A?xoBpL>MD1A)381YOi_+T|*(pNO{mB~NU5=(Ut{DLhjxew)?(;3&ZV{q#PQQ%aW`ZA;Jk)Y*N5%z"
    "zn>i-Q^UUY>aU|7*kgLzp41<;XGe?;Yqf&~iA*NIFC$I?}4<99Mp;p$fE%}P0;DA7oS_YK7DbkG2(d}v_Wdw|ppIW|6;tEG*"
    "b8550pWVNzYd_PFo#X@B#d6+N&FE+A<5%6VsY;QhZ#&b$E0Sc+-Eu}Am$=Uu?4uJ_@ur4^qM-+lD3WS=808WtQzJ8#73~1+#"
    ">yK3PJd5L^q^(yG3aZwxFT#=j-<RV+O1EaGzRc=a2S_Ep6mB7&;0)U2uZ=SJ@xV>gJ@YIh{1?)U8t&a~=*o|;0^izlQ~T&PI"
    "PpDDVvmRo!^<6{I{BUhZ_!t*?Z-`&4RkPoLS%$*>*i8UX$g>gPCeh#_f3ZhC$|;;O|FH7C8Qm|Og@NJ*g&%xAUQvyTq*V{O?"
    "@ppRbj9ry&vfu^Y;&hrn!1h==#NPCZwt}zCwZF4~gW}Y3pPo#m~;~8yaO<h=Edny`_1}t*WzMx6j8f<furBN9|w9wICl5lu*"
    "rOTPFfiSg>~-{#HEy%Erjt(IWbBW1REvhiNrsJoMjDyJeGV--yCZ&pmXWm<@7YE_E_=WjRBALK$@ibV2CIOW2BdNbl;}4VBm"
    "&%b(6UFv>|Qv-SHS=)rtlju)B``72~@_V-oQ!r;L!8}fJzaxQv);&%-6_{xdv0VXJ5L1O*#h3oMMp<c+&y}Y^d>g3-V=;z44"
    "15dF!DLJ&B_jsthe}(bn_Rz0z!6LC*m6v7qTaM&tFFh3@2}?P^Xjr2hazgY1=-?ea*Z3ZuWgjdbKFx7|=*6PHSJ&}itB#Zlv"
    "Rh|Y)-;+UP1%;47DKep*sT5;nM1DK;QGZTOGyoUtU1Wa+3S-War90gBTe-g#y+(zO(H8`hf2%mF@Vy#jWmG;;@Tjd{kK`<Aw"
    "d!MjN3;8|4m3;!iq3~;~$%qy^a_QnIlZ;?VvSLj*OeS(dxoN`j9P6QK@g0)58L!uWqiws~zNA>x@0?r0nA9tezY)6B?n!SrG"
    "qOw=G5`;Z}dL2NT-s7MV&7lRf#Z7tFOe{O+7lQfs%im+z^$q)@$QL`_C-BWV}Ioqu1l;aIhKj@3t%SF+I_C~_yld0^!r2m`G"
    "H-jzKn8YIv@gCqg7P_fMwRRhJq!B?lfR18m5dqoHpFy>`38hO2+VUr;F^yVCv(adFf@O`-JT|B6E>3)zP499lc)lVR}WgZXy"
    "{M;<ymxr+#Uzf-6|Nbhnz}f3A1|p?lUm;HaDax3Yh|l-6P?POlz2hQ%kr?v;R6)gBB)(qjN~kS5N}1m`vwJlDbi_2)vT73=4"
    "sE%`RQ*J!5fme^Bo*7azvZ2OsH)62wg^pX+GW%*M=>6`O|W%Hj1aN%gO_3vn6TgVc5cmgxqrU@SQ@xeJfMXm_y~c?d{Z$=+S"
    "vnV;%`??)A@R-t}1uKMI^dkTosep@eRGJSU)wB)Xww?JVILU=xJyr=NX#~Sw;x5AoyEmBW*}PHAdtOkR&yr9mv_FIxhOY?2&"
    "MvB1gm_p;a<dEH4iG!V5$y8v16PW01HI<t0Z=IAJQmie}z)t3UCI6yu+Z+7I%5Lj4(8@TrYm5fVR{+Cgckz2TBRGl3*7MmY|"
    "|S>u|ESZCSY`fc!wFV#R0P4O6x4=~ObGt4kcrdliz6m~*jay(KIIn@Q?c}@g>DEu8Xju4!mNdkGWfDUnuTwk295|`qJ?P+au"
    "d^)GWK}N~4ke|$s2MxN+p<y&b6jQA^8R{9<cAHWzIh56I8zt)TFrk88cKNs>hhesJW8LhoX$+q`vAnxj2Bej}K_{dqP!fGoF"
    "3k7tsd3fWGqgCA;{lq7p%zZ+9E}Euy&8#t7%8Ai7({7oag4|MYsEz*8ggP7XD8gseT{YwTAn8QjI!tI7nl8lNd+6hhVy!4to"
    "M)Ek7;fh1@90hYHTSczFHvNp$ty<yYRgB#rQ61_#5@kdg7kw$)k=V?ik}hyXiH)JN`BeE3XCrGfbAz2pF)vArCIdIde9WC)4"
    "2G(0ai>`^M~w<AQ`&Ot?C+kS*RxhTFtnj+GsMEpcrEPPJNLTjq5|N0B13H3nlkqQi#R;S@=^xZbi!W{8t>HtKy@6FR4?mTiZ"
    "nn7hSV?raV~sJ506A0@08AIin9-%N%%B2F?M3)a|SpEz+u+q%%x#S8&QLWyKM6B#M~+7__eyTs1{6o7=Sc5}s}UY?-g>hnQm"
    "Lc+Me(m4Pps?m0<uaas1FyC;A$Q6A`?Rt%>);;3B91=&zSg+;Q+YG+LPMxnHAd6AI?mtUc-YVx`+@E>jVl*VGgd2NiMP&_se"
    "hyy<%YR=20whE!%eC71$r9Q|8<;yWJp`Tc*JzQ7xIdv)MIf(0NI9qn({UU+rMDY7f*h9c&J1t95d=_MQ?56^FB|w=H~bHYe{"
    "m!B<yOwXd?0E{<J3pR0%SiQNZ$F=z+Rb$+#;yXz1)bu%}#s7<DVbF8zl#xdogGFCrkZN!lcW>!6y=XMv(JIuAO_7Ktofku(M"
    "jYsg?sw5uK0f+Tu}eh4$?6-dc}(T*YGDfAq^P*zJ#@%$IEe@a^aqXu)vU=2nx2vjd-9kbCrO+z1X4%}Rmk-evW=xP&Pp!82u"
    "3(C`D%`_{A<q~fWjtPN(S{(#`12^GODAKLP+qX_0Vhl<7ayLFavR2OmNEeDuWJmS%Bh;#aGXi*h9H7@k?N)Bt3vhP@`7T4EH"
    "9s4m)PYGu&{(ZS3ZQzU}1f5U!sLP`*qPLG#TN3Xz!+Ew}yd2w<)lH(R2@p+fre}Afx+LRBQ5<0Q29ZX-7WdIP%EJF0Zru+cU"
    "p2JXDBes3*{~ORA!NEltEdE>_l2m(DiN6O`}tdH_}Dm%_=}vK#?Z<Ryz;rko}2oZJ_2+dGU}{$;n=fma)%a4R>RD@c(qldw&"
    "JO@!2R^8{j^fz=~IE#>`L*rg?+a0w{`{`8;<+^Qad?{*c9;d&cnNc#pucGzptoJhxhxkAx>~vh3A1UFt$<vkE<sv?1rBa$ie"
    "BWdrxM{AzdK}q&|JJUvGCe@xiW%*Ipk}YA(m`unk*P>{kD_CJR$(#)X{j58iG2y1^WA3!WeE$RA$`O_9F8yF>y(7-vb%^TV0"
    "#19yqCTq#;45ddN1y#8}=;>O5ldo@58&w<FjiPM;ce(B?Ht`U?z>~Q2zI~FaZX&-G9jbUKALZcQycO>!1ajo6g@#`zr5|`4%"
    "z}F`SY$E>(uu}P^^_HB%1=i?mNsOiuj1#36^&G9Wpwcf&P)*<0i6+qyYEGx<NQT(4c)>Jd-23|E<Fg7=@0gOx6d=>4q{m3(#"
    "XBR&QV~czkA?;MZqYfCL?XKfoh>bH<S?a1CLz<XQ`4)`kDAGGY|QM>9}y)e#v`dAH&7~jn#;LO@UMxKw`6Q@2J<$ox+nN<dW"
    "_bHj|T1{^EoE>>lwkGLhAm5*lsGLdyG2PdOlNj&xb=)pVd)(gT%HbD6?7bucd?YSNKtv9>QG)<(`npM9xAz;zvb3Ti!_1y{8"
    "#LkN0<*qGAu38#zuG8WC5y1v3Y_6W3vM;3(d3u?P&&3+5whBpG7!#plSMS5T(k##`5gKkPfE|LEM#XP=LF(xkL=YNo>f3hBF"
    "VmH1FNhm0E`6KwUK<R%qeJr7--m#Qn2j?NfZ>G4Px;awax-R6Tfw{FOFg)<qef<-PPg-0KC@*A+T^NF1Y8%^{|)d#liYIcHS"
    "Tt6#Fo-%`3ngH6EusXW#wl2?k<|~IDfAC*#Q%W7Up(CNptlIQ8G~v1}XorMD%O=&W7aeELPnoNe4wV-NS4_4VRW5Zn-Ik^HX"
    "$To<q+gw}P(}OjTJ;4Fj#Lt1{4qyPc6An(_bpaqmw_$E#VHR<`MQ?^6?|GgRUG-WhIkt|BnLAlxj{a(8AX)dY)k7Xr;w14Wf"
    "=<tYdl+mzHR)GO&*oR8pm_=p7;*i-f`;mJK=S(Q?FGh7t{7&OKS7_<NeY+JJ9~(2?wc$iVW9cWZMehtFbK*dc-hU9h$~8L*S"
    "}P*fIu9q5PRI!e`zzAVkVV1=moQTMRQWm!cAQ^iYPF-Mq+}4yO!6Dp6It%{}<iZBJjjuDwzM2_$5Br9Y%hl#@Ko=cFve;re-"
    "uH)CKzSHRTSZCod`dsmaRA(&+XoeB62{E)S0yjX5<Bsc=BbD7Q}u@#aXu>=iv4U9y7XcF6a$h3L20GPxn{w^7roVQ|ogQGGH"
    "vH=!e(bSez-8zyDr&JT9J*R%s|8Pob(9S*uysE6=0wi4^EQY$x>u64AgUgRAu2<v=Oh!;0cgMgzvRe2+50~Nb`uRZEt?|Nw#"
    "O~NyYeF=Xuk?<b0&eCyB;9~H71&6eK$7+ocEH9ncGmmqQ!tYeK5D+l2di7)(Vi`V`(N%Ee$SxfFbnr7cO`Z`n$C9YIoG|HPp"
    "<CcMwDyAN^FVROXsC!H=)sPs}e8yPLF~4sj+}j`)hfM<q6Eows614hL%WjNW9axbkOy7pD3#$-Wc-w6a7{z%MY`J#y+k3#>x"
    "g*j<RW13JQdlY*J}C=JOpI25efd)Ah$0qpiZTI7KCC&~qD}#m!t!r{inxmz{w+fD0{Jk7z;4E{V-s#j9=P_9xxd)V1$A*T}n"
    "Z(aR*z<g}_-5EGao{9^?M-cmj0tCPbb<1q@4W}~Bb{VcrTQO#b6tt6-pnE&il<5J4D_QBHKa}(DZ{1IS7y)+MdJJN?9kEvdC"
    "3&myMuRUhPK9;e+zb@g(zk76&?(OqQN!_|q>zfV;J<X)HA)<E5m9jO;*l}l~3dAhh^+40uq{V=(;a4H$eTNLswZgfaRp4U&@"
    "l~!1!>;wErHwjIm6G8;VzrBLGX)jyX2i>$)~>ADzO+pZLhnv&F@F"
)


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawModule(
            code=_CUDA_SRC, options=("-std=c++17",),
            name_expressions=("closed_loop_kernel",)
        ).get_function("closed_loop_kernel")
    return _kernel


def closed_loop_batch_gpu(Yb, Ub, Vb, modes_list,
                          y1, y2, uv_m, y1deq, y2deq, uvdeq):
    """Batched GPU closed loop.  Yb (B,H,W) / Ub,Vb (B,H/2,W/2) int16 padded;
    modes_list: list of per-image mode dicts (numpy).  Returns
    (y_dc, y_ac, uv_lv) numpy int16 arrays with leading dim B."""
    B, H, W = Yb.shape
    HH, HW = H // 2, W // 2
    mb_h, mb_w = H // 16, W // 16
    n_mb = mb_h * mb_w
    dev = cp.cuda.Device()
    is_i4 = cp.asarray(np.concatenate(
        [m["is_i4"].astype(np.uint8) for m in modes_list]))
    i16m = cp.asarray(np.concatenate([m["i16_mode"] for m in modes_list]))
    uvm = cp.asarray(np.concatenate([m["uv_mode"] for m in modes_list]))
    i4m = cp.asarray(np.concatenate([m["i4_modes"].reshape(-1)
                                     for m in modes_list]))
    q = [cp.asarray(np.ascontiguousarray(a, np.int64))
         for a in (y1.q, y1.iq, y1.bias, y1.zthresh, y1.sharpen,
                   y2.q, y2.iq, y2.bias, y2.zthresh, y2.sharpen,
                   uv_m.q, uv_m.iq, uv_m.bias, uv_m.zthresh, uv_m.sharpen,
                   y1deq, y2deq, uvdeq)]
    y_dc = cp.zeros(B * n_mb * 16, cp.int16)
    y_ac = cp.empty(B * n_mb * 256, cp.int16)
    uv_lv = cp.empty(B * n_mb * 128, cp.int16)
    rY = cp.empty(B * (H + 1) * (W + 1), cp.int16)
    rU = cp.empty(B * (HH + 1) * (HW + 1), cp.int16)
    rV = cp.empty(B * (HH + 1) * (HW + 1), cp.int16)
    flags = cp.zeros(B * n_mb, cp.uint32)
    kern = _get_kernel()
    kern((B * mb_h,), (1,),
         (Yb, Ub, Vb, is_i4, i16m, uvm, i4m,
          q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9],
          q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17],
          y_dc, y_ac, uv_lv, rY, rU, rV, flags,
          np.int32(B), np.int32(mb_h), np.int32(mb_w),
          np.int32(H), np.int32(W)))
    out = cp.concatenate([y_dc.reshape(B, n_mb, 16),
                          y_ac.reshape(B, n_mb, 256),
                          uv_lv.reshape(B, n_mb, 128)], axis=2)
    o = cp.asnumpy(out)                 # single D2H transfer
    return (o[:, :, :16], o[:, :, 16:272].reshape(B, n_mb, 16, 16),
            o[:, :, 272:].reshape(B, n_mb, 8, 16))


_mode_kernel = None


def _get_mode_kernel():
    global _mode_kernel
    if _mode_kernel is None:
        _mode_kernel = cp.RawModule(
            code=_CUDA_SRC, options=("-std=c++17",),
            name_expressions=("mode_search_kernel",)
        ).get_function("mode_search_kernel")
    return _mode_kernel


def mode_search_batch_gpu(Yb, Ub, Vb, y1):
    """Fused mode search on padded planes Yb (B,H,W) etc. int16.
    Returns the raw bundle (i16_mode, i16_score, uv_mode, sse4, mb_w, mb_h)
    bit-identical to the vectorized gpu_modes_pass_batch(select=False)."""
    B, H, W = Yb.shape
    HH, HW = H // 2, W // 2
    mb_h, mb_w = H // 16, W // 16
    n_mb = mb_h * mb_w

    def borders(P):
        b, h, w = P.shape
        out = cp.zeros((b, h + 1, w + 1), cp.int16)
        out[:, 0, :] = 127
        out[:, :, 0] = 129
        out[:, 1:, 1:] = P
        return out

    bY, bU, bV = borders(Yb), borders(Ub), borders(Vb)
    from . import vp8_tables as _T
    fc_i16 = cp.asarray(np.array(_T.FIXED_COSTS_I16, np.int64))
    fc_uv = cp.asarray(np.array(_T.FIXED_COSTS_UV, np.int64))
    i16_mode = cp.empty(B * n_mb, cp.uint8)
    i16_score = cp.empty(B * n_mb, cp.int64)
    uv_mode = cp.empty(B * n_mb, cp.uint8)
    sse4 = cp.empty(B * n_mb * 160, cp.int32)   # 16*255^2 fits int32 easily
    n = B * n_mb
    kern = _get_mode_kernel()
    kern(((n + 127) // 128,), (128,),
         (bY, bU, bV, fc_i16, fc_uv, i16_mode, i16_score, uv_mode, sse4,
          np.int32(B), np.int32(mb_h), np.int32(mb_w),
          np.int32(H), np.int32(W)))
    return dict(i16_mode=cp.asnumpy(i16_mode), i16_score=cp.asnumpy(i16_score),
                uv_mode=cp.asnumpy(uv_mode),
                sse4=cp.asnumpy(sse4).reshape(B, n_mb, 16, 10),
                mb_w=mb_w, mb_h=mb_h)  # sse4 int32: half the D2H bytes


_defilter_kernel = None


def png_defilter_batch(raw, rows, rstride, W, bpp):
    """raw: (B, rows, rstride) uint8 filter-prefixed inflated rows.
    Returns (B, rows, W, 4) uint8 RGBA (RGB expanded with alpha=255)."""
    global _defilter_kernel
    import cupy as _cp
    if _defilter_kernel is None:
        _defilter_kernel = _cp.RawModule(
            code=_CUDA_SRC, options=("-std=c++17",),
            name_expressions=("png_defilter_kernel",)
        ).get_function("png_defilter_kernel")
    B = raw.shape[0]
    d_raw = _cp.asarray(raw)
    out = _cp.empty((B, rows, W, 4), _cp.uint8)
    _defilter_kernel(((B + 31) // 32 * 32,), (32,),
                     (d_raw, out, np.int32(B), np.int32(rows),
                      np.int32(rstride), np.int32(W), np.int32(bpp)))
    return out
