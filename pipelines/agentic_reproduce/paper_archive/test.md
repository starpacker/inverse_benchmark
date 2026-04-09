

# Coherent diffractive imaging simulations  for wafer inspection of periodic structures 

Paolo AnsuinelliaHojun Lee, Wookrae Kim Junho ShinYasin Ekinci,a [Ta and lacopo Mochi 

Villigen PSI, Switzerland 

Hwaseong-si, Republic of Kora 

Samsung Electronics Co., Ltd., Core Technology R&D Team, Mechatronics Research, Hwaseong-si,Republic of Korea 

## ABSTRACT

.Background: One of the primary challenges in semiconductor metrology is highresolutioninspection of lithographic patrns.Although the patrns can be rlatively complex, periodic structures are found in many layers of modern devices, and their metrologis fundamntl.Inte context ofhybrid bondingaperiodi aray ofcopr pads within a wafer is used for electrical connections, and metrology methods are deployed to characterize their topography. Although conventional methods include,e.g., atomic force microscopy and scatterometry, we discuss coherent diffractive imaging (CDI).
Aim: We aim to study CDl for the metrology of the copper pads used in hybrid bonding. We aim to demonstrate that CDl is a good option for this problem, especially given the phase sensitivity, which may be advantageous when a sample with slight topographic changes is inspected.
Approach: CDl is employed for the metrology of a copper pad array. We model the sample as a reflection function.The impact of the copper pad topography on the data is introduced by a phase term that accounts for the optical path length difference induced by the recessions in the copper pad with respect to the surrounding layer.In addition, standard phase retrieval algorithms are modified to incorporate information on the reflectivity valueof the substrate/materials that surround the pad array.Results: Simulations show that the proposed algorithms can drastically improve imaging. Features are better resolved, and the loss function associated with the proposed methods can decrease up to 2 orders of magnitude with respect to standard phase retrieval methods.Thisimprovement is more substantial for an Airy spot illumination, and it is milder when a structured illumination is employed.Conclusions: We present a study of phase retrieval algorithms applied to the metrology of copper pad topography for hybrid bonding. We demonstrate that by including a prioriinformation in the update function of the object, a better estimation of the recession of the copper pads can be achieved, leading to an improved metrology by CDI.
© The Authors. Published by SPIE under a Creative Commons Attribution 4.0 International License.Distributionor reproduction of this work in whole orin part requires full attribution of the original publication, including its DOI.[DOl:10.1117/1.JMM.24.4.041403]
Keywords: hybrid bonding; advanced packaging; Cu pads; coherent difractive imaging; phase retrieval; wafer metrology 

Paper 25063SS received May29,2025;revised Jul.26,2025; accepted Aug.7,2025;published Aug.28,2025.



## 1 Introduction 

graphic.ed in many layers of modern devices. With the growing interest and adoption of hybrid bonding for wafer-to-wafer (W2W) or chip-to-wafer (C2W) interconnects, their metrology is becoming increasingly important. Copper pad arrays,used to form lectrical connections in hybrid bonding, are examples of such periodic structures. Hybrid bonding enables vertical integration of multiple layers of chips through direct copper-to-copper and dielectric-to-dielectric bonding.This eliminates the need for soldering bumps and enables high-density interconnections, better signal integrity, and pitches below 1 m.



Metrology of the Cu pads'surface is necessary to ensure that the nanotopography of the copper pillars is compatible with the bonding process. The metrology method chosen for this step needs to accurately characterize a large area in a suitable amount of time to satisfy throughput requirements. Techniques such as atomic force microscopyl and scanning electron microscopy 2can be used for the purpose; however, it is interesting to evaluate metrology alternatives with potentially higher speed, throughput, and accuracy.



Lensless imaging techniques have revolutionized the field of microscopy by enabling highresolution imaging without the need for conventional lenses.3 Among these methods, the hybrid input–output (HiO) algorithm4 emerged as a cornerstone for phase retrieval in coherent diffractive imaging (CDI). By iteratively enforcing constraints in real and reciprocal space, the HIO algorithm reconstructs sample images from their diffraction patterns. However, its performance can be hindered by slow convergence rates and suboptimal accuracy, particularly when imaging samples with complex structures.



Ptychography,  another prominent lensless imaging technique, has gained attention for its robustness and accuracy. By scanning overlapping regions of a sample, ptychography achieves high-resolution reconstructions with enhanced stability against noise. However, this method comes with increased computational demands, which makes it less practical for scenarios requiring rapid imaging. Moreover, under certain conditions, ptychography does not perform well with periodic samples.6 Although other solutions can be adopted for such metrology samples,7,8it is interesting to study and evaluate the phase retrieval methods that can be applied to periodic targets, given the challenge they can pose toward this class of computational methods.

When illuminated by a beam, a periodic structure diffracts light in a discrete set of orders. If the period of the structure and the numerical aperture (NA) of the illumination allow for it, the diffracted beams can overlap at the detector, and they can interfere. By scanning the periodic structure with respect to the ilumination, the phase difference among the overlapping orders can be determined, and the phase can be retrieved.9,10When the period of the structure is too small,the orders are separated by an angle larger than the illumination cone, and the diffraction patterns remain almost unvaried across the scanned area. Because ptychography exploits translational diversity, its performance is challenged.



In this paper, we will focus on lensless imaging as a candidate method for the metrology of copper pads in hybrid bonding. In particular, we apply the algorithm presented in Ref. 11 and extend its application to ptychography. Unlike Ref. 11, the sample presents a phase profile that is not constant across the grating due to the recessed topography of the Cu pads. The algorithms are modified to include a constraint on the reflectivity of the sample in the regions where the Cu pads are not present.1² We call these algorithms reflective-prior input–output (rpIO) and reflectiveprior ptychographic iterative engine (rpPIE). The proposed modification improves both the efficiency and the accuracy of the reconstructions. Through simulations of a periodic array of pads,we compare the performance of the rpIO and the rpPIE against that of the conventional HIO and ptychographic iterative engine (PIE) algorithms. Our results demonstrate that the joint use of a structured illumination 13and customized algorithms results in a faster and more accurate object reconstruction than for the case of an Airy spot illumination and standard algorithms. We stress that although these computational methods are applied here to a specific target, they are generic and can, in principle, be applied to various samples.



ptychlminations.Sectionconcludewihtheimlicatioofofndiganpotentialfuture directions.



## 2 Algorithms Description 

TheHIO aloim4iaiativmtoudtcotructephaseoajcivenhe measured data and someknowledgeof itssupport inreal space.Givensupport , aD coordinate vector $\mathbf{r}=[x,y]$ , and a complex-valued object $O(\mathbf{r})$ defined within the support, the update rule for the object at iteration $n+1$ reads 



$$O_{n+1}(\boldsymbol{r})=\left\{\begin{aligned}&\mathcal{H}^{-1}\left[\sqrt{I(\boldsymbol{k})}\frac{\mathcal{H}(O_{n}(\boldsymbol{r}))}{|\mathcal{H}(O_{n}(\boldsymbol{r}))|}\right],&\mathrm{f o r}\boldsymbol{r}\in S\\ &O_{n}(\boldsymbol{r})-\beta\mathcal{H}^{-1}\left[\sqrt{I(\boldsymbol{k})}\frac{\mathcal{H}(O_{n}(\boldsymbol{r}))}{|\mathcal{H}(O_{n}(\boldsymbol{r}))|}\right],&\mathrm{f o r}\boldsymbol{r}\notin S,\end{aligned}\right.$$

where I represents the intensity distribution of the diffraction pattern, β is a feedback constant,and H is the propagator. The HIO uses feedback in the real space domain to drive the reconstructed object toward the ground truth.



In this paper, we consider phase retrieval from far-field data. The far zone is defined by a condition on the propagation length, z.For a given scatterer dimension (d) and a fixed wavelength (), the following relation must hold 



$$z\gg\frac{2d^{2}}{\lambda}.$$

When $\mathrm{E q.}$ (2) is valid, a source fild $U(\xi,\eta)$ is associated with a far-field expressed by 

$$\frac{e^{-ikz}e^{-i\frac{k}{2z}(x^2+y^2)}}{-i\lambda z}\iint U(\xi,\eta)\exp\left[i\frac{2\pi}{\lambda z}(x\xi+y\eta)\right].$$

Equation (3) states that the field $U(x,y)$ measured at a detector is proportional to the Fourier transform of the source field,$U(\xi,\eta)$ |, evaluated at frequencies 

$$f_{x}=\frac{x}{\lambda z}\quad f_{y}=\frac{y}{\lambda z}.$$

Therefore, the propagator H in Eq. (1) corresponds to a Fourier transform. The algorithm described in Eq. (1) can be applied to a generic sample with finite support; however, here we are interested in finding ways to improve it for applications to lithography targets.

Patterns in semiconductor device layers can often be modeled as binary structures with different transmission and reflection properties. In general, we can assume that a planar structure of a given material will have a uniform reflectance and phase response, whereas a curved surface will deform the reflected wavefront and locally modify its phase. The algorithm we propose is designed to reconstruct the phase and topography of a sample with periodic structures, an array of copper pads, by constraining the reflectivity of the surrounding planar $\mathbf{layer}^{11}$ 

$$O_{n+1}(\boldsymbol{r})=\left\{\begin{aligned}\mathcal{H}^{-1}\left[\sqrt{I(\boldsymbol{k})}\frac{\mathcal{H}(O_{n}(\boldsymbol{r}))}{|\mathcal{H}(O_{n}(\boldsymbol{r}))|}\right],&\text{for}\boldsymbol{r}\in S\\ \beta O_{n}(\boldsymbol{r})+(1-\beta)R_{p},&\text{for}\boldsymbol{r}\notin S.\end{aligned}\right.$$

The nomenclature in Eq.(5) is similar to the one used in Eq.(1): n is the number of iterations,$\mathbf{r}=[\xi,\eta]$ iseo $I(\mathbf{k})$ represents the measured diffraction intensity,$O_{n}(\mathbf{r})$ is the compexobjctfunctio at atio $R_{p}$ is the reflectivity of the regions of theampleutsidethesuppor,istheedbackconstan,andSisthe support.In this case, the support is an arrayof ones and erosthatdefinesthegeometryand periodicityof the pads.



i $P(\boldsymbol{r})$ ,a feedbackotat,n an xit wave $\Psi(\boldsymbol{r},\boldsymbol{R})=P(\boldsymbol{r}-\boldsymbol{R}_{j})O(\boldsymbol{r})$ , which is eventually We inrod ce apre c $\Psi_{c,n}$ .$n+1$ , reads 

$$\left\{\begin{aligned}O_{n+1}(\boldsymbol{r})&=O_{n}(\boldsymbol{r})+\frac{|P(\boldsymbol{r}-\boldsymbol{R})|}{|P_{\max}(\boldsymbol{r}-\boldsymbol{R})|}\frac{P^{*}(\boldsymbol{r}-\boldsymbol{R})}{(|P(\boldsymbol{r}-\boldsymbol{R})|^{2}+\alpha)}\times\beta(\Psi_{c,n}(\boldsymbol{r},\boldsymbol{R})-\Psi_{n}(\boldsymbol{r},\boldsymbol{R}))\\ O_{n+1}(\boldsymbol{r})&=\beta O_{n+1}(\boldsymbol{r})+(1-\beta)R_{p},\qquad\mathrm{f o r}\boldsymbol{r}\notin S.\end{aligned}\right.$$

### 2.1 Simulation Parameters 

The object we simulate is a $15\times15$ set of pads. Each pad in our simulation occupies 22 pixels of the object matrix, and the grating has a duty cycle of 0.6. The wavelength,λ, is 13.5 nm. For the simulations, we use a pixel size of 27 nm, close to the resolution of our experimental setups.The critical dimension (CD) of the grating is $\approx600$ nm,whereas thepitch is~800nm.A sketch of the object is shown in Fig. 1, whereas the values of the materials n and k are reported in Table 1.15



For the simulations, we consider a reflective object. The reflection function is computed by the transmission matrix method, and we evaluate the phase difference introduced by the concavities of the Cu pads with respect to the flat surrounding as 

$$\Delta\Phi=4\pi\frac{h(\xi,\eta)}{\lambda}.$$

We assume that the concavity of each pad can reach a maximum value between 1 and 3 nm.For the maximum value of 3 nm, the phase modulation introduced by the concavities is 2.8 rad.Therefore, considering this recession depth range prevents the occurrence of phase wrapping while maintaining a realistic range of values for the simulations. Although the model we adopt 

<div style="text-align: center;"><img src="imgs/img_in_image_box_315_818_869_1133.jpg" alt="Image" width="45%" /></div>


<div style="text-align: center;">F $h,$ oftei pad is selected in the interval [1,3] nm. We study an array of $15\times15$ pads.</div>


<div style="text-align: center;">Table1Materials n and k values taken from the CXRO database.15 </div>



<div style="text-align: center;"><html><body><table border="1"><thead><tr><td>Material</td><td>n</td><td>k</td></tr></thead><tbody><tr><td>SiCN</td><td>0.9791</td><td>0.0059</td></tr><tr><td>SiO2</td><td>0.9780</td><td>0.0108</td></tr><tr><td>Si</td><td>0.9991</td><td>0.0018</td></tr><tr><td>Cu</td><td>0.9625</td><td>0.0613</td></tr></tbody></table></body></html></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_313_10_886_554.jpg" alt="Image" width="46%" /></div>


<div style="text-align: center;">Fig. 2 Adopted probes, an Airy pattern, and a speckle field, in magnitude (a) and phase (b). We scan the probe in steps of 40 pixels. The insets outlined in red show a magnification of the respective region in the speckle field. </div>


is not rigorous, we stress that it has been experimentally proven that actinic ptychography is able to resolve optically thick objects in reflection mode.16,17

As stated in Sec. 1, we test the phase reconstruction using different illumination patterns,which consist of a structured illumination (a speckle field) and an Airy spot. The structured illumination and the Airy spot, in magnitude and phase, are shown in Fig. 2. In all cases, the illumination propagates perpendicularly to the sample plane, and Gaussian noise is added to the diffraction patterns. The reconstruction obtained by the standard HIO algorithm is never shown because the algorithm did not converge to a meaningful solution. In the case of ptychography,we scan the probe to ensure an overlap of ~75% among successive probe positions.

### 2.2 Phase Retrieval in the Fraunhofer Region 

In this section, we show the result of the phase retrieval from data collected in the far-field. An example of an object and its respective Fourier diffraction pattern is illustrated in Fig. 3.Notaly,the diffraction pattern in Fig. 3 indicates that the main diffraction peaks are separated at the detector. The impact of the overlap among the main diffraction peaks and lack thereof in both phase retrieval and scatterometry has been discussed, e.g., in Refs. 6, 10, and 18. As mentioned in Sec. 2, the sample we simulate is a 15 × 15 array of copper pads having a periodicity of ~80 and 

<div style="text-align: center;"><img src="imgs/img_in_image_box_134_1180_1027_1427.jpg" alt="Image" width="72%" /></div>


<div style="text-align: center;">Fig.3lllustration of an object in (a)magnitude and (b) phase.Panel (c) is a noiseless diffraction pattern, obtained by a plane wave illumination,and in panel d), we show a oom on the cntral art of the diffraction pattern [red square in panel (c)]. </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_155_14_1028_667.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Fig. 4 Reconstruction of the target in the Fraunhofer region. Panel (a) shows the outcome of the rplOin magnitude and phase, as wll as thePIE (b) andthe rpPIE (c).Paeld) istheground truh.For all cases,a cross-section of the object across the red line shown in the figure is given in panel (e). The error is shown in panel (f). </div>


a CD of ~600 nm. We include uncertainty in the dishing of the copper pads, and we assume that the recession in each pad, with respect to the surrounding planar layer, can change from a minimum of 1 to a maximum of 3 nm. The reconstructions obtained for this sample are shown in Fig. 4, where we notice that both the rpIO and the PIE show poor reconstruction quality, whereas the rpPIE succeeds in achieving a beter reconstruction. The difference in the reconstruction quality is visible both in the cross-section plot and the error plots. Particularly, the latter one demonstrates a decrease of 2 orders of magnitude in the error evaluated by the rpIO with respect to the PIE. Figure 5 shows a three-dimensional rendering of the phase of a single Cu pad reconstructed by the rpPIE from far-field data and its discrepancy from the respective ground truth. We notice, all over the region of interest, a discrepancy in the phase of a fraction of a radian, with the most substantial difference toward the bottom of the parabola described by the Cu pad phase.Although further algorithmic efforts could, in principle, mitigate this discrepancy, it is interesting to study the impact of a structured illumination on the reconstruction quality. This is because, as noticed elsewhere,6 the estimation of the phase can be improved using a structured illumination (cf.Fig.2), asshowninFig.6.AcomparisonbetwnFigs.6and4revealsthatallthepresented phase retrieval algorithms perform better when a structured illumination is employed. Although Ref. 6employed a vortex beam to illuminate the sample, here we simulate a speckle field. Such a field can, in principle, be generated by using a Fresnel zone plate with engineered zones to introduce perturbations in the wavefront,19or by, e.g., transmissive/reflective phase structures.20 The improved reconstruction may be attributed to the incrased illumination NA.In Fig.7, we show a three-dimensional rendering of the phase of a single Cu pad reconstructed by the rpPIE from farfield data. A comparison between Figs. 5(c) and 7(c) reveals a substantial improvement in the reconstruction of thecopper pads when a tructured illumination is mployed.We notice that many points acrss ofist  apachlueoftewchiet,nd that the bottom of the parabola, a particularly sensitive point in Fig. 5(c), is now well reconstructed.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_230_6_998_628.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">ef a sinale pad in the Cu pad array  spot in Fig. 2 and the rpPlE algorithm. (b) Ground truth. (c) Phase error (top view). The average  value of the phase error through the Roi is ${\sim}2.6e^{-2}$   </div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_189_767_1047_1397.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Fig. 6 Reconstruction of the target in the Fraunhofer region using the speckle field in Fig. 2. Panel (a) shows the outcome of the rplO in magnitude and phase, as well as the PiE (b) and the rpPIE (c).Panel (d) is the ground truth. For all cases, a cross-section of the object across the red line shown in the figure is shown in panel (e). The error is shown in panel (f).</div>


## 3 Experimental Setup Concept 

simulation study is a Cu-pad array with submicron pitch.The goal is to characterize the topographf i setup with a resolutionof at least onetenth of the pad size,which,in this case,corresponds to 40nm.This resolution enables adetaild amplingof the pad surce.This resolutionhas aleady been demonstrated using extreme ultraviolet (EUV) with CDI setups.21,22

At normalincidence, EUV liht isabsorbedby mostmateials,includinglicoxd an copper. This requires grazing incidence ilumination to increase the sample reflectance and guarantee a good signal-to-noiseatioin theactiondata.However,in thiseometry,theconial diffraction effect is predominant, and the diffraction orders are mapped onto a conical surface.The algorithms discussed in Sec. 2 make use of the fast Fourier transform algorithm to simulate the propagation of the EUV beam between the sample and the detector. This requires remapping the diffraction data to correct for the conical distortion. This can be efficiently done with simple interpolation procedures well described in previous works.23,24

REflective Grazing Incidence Nanoscope for EUV (REGINE)25 is a microscope developed for synchrotron-based EUV coherent diffraction imaging that could be used to collect the diffraction data to be used with the algorithm described in this work. A 3D model of REGINE is shown in Fig. 7. The microscope is equipped with an EUV detector with a $2048\times2048$ pixel array at a distance of 62 mm from the sample. As the size of the detector is 30 mm, the collection NA is 0.24, and the resolution is ${\sim}35$ nm.



<div style="text-align: center;"><img src="imgs/img_in_image_box_141_694_1056_1411.jpg" alt="Image" width="74%" /></div>


<div style="text-align: center;">oee) of a sinalepad inthe Cpda  ${\sim}4.1e^{-4}$ h.)er. The average value of the phase error through the ROi is  </div>


more than $10^{12}$ p a meantime,Vo silciicto EGEcouldb usd to verif exile ofi i vously in Sec. 2.



## 4 Conclusion 

Hybrid bonding is necessary for high-density interconnections in semiconductor devices. Cu pads, embedded within a wafer, enable the connection of die-to-wafer or W2W without the need for soldering. The metrology of the Cu pads is necessary to ensure that their topography is appropriate for bonding as the presence of irregularities can compromise successful bonding. In this paper, an approach based on lensless imaging has been proposed and studied by means of simulations. We notice that ptychographic methods reconstruct a given sample transmission/reflection function. Therefore, to improve the estimation of the sample phase, one might think to formulate an update rule that enforces a constraint on the sample reflectivity in those regions where this could be estimated. Based on this consideration, we propose two algorithms, which we call rpIO and rpPIE. To characterize the performance of the proposed algorithms with respect to standard methods, we have considered the problem of phase reconstruction from far-field data generated by illuminating a periodic array of copper pads (Fig. 8).

The proposed study shows an overall improved phase retrieval, particularly when the proposed methods are applied to a dataset acquired by illuminating the sample with a structured illumination. Notably, the proposed methods can be applied to any sample geometry, provided the reflectivity in a given area can be estimated. This condition is often met in semiconductor device wafers and photomasks, where material properties are well characterized. By improving the reconstruction accuracy of current lensless imaging techniques, rpPIE and rpHIO may thus help establish CDI as a viable solution to some of the challenges in high-volume manufacturing semiconductor metrology.



<div style="text-align: center;"><img src="imgs/img_in_image_box_148_877_1041_1250.jpg" alt="Image" width="72%" /></div>


<div style="text-align: center;">Fig. 8 (a) REGINE 3D model. The microscope is installed in an ultrahigh vacuum chamber and operates at a pressure of 10- mbar to avoid any absorption of the EUV beam by air. The EUV beam used for REGINE has a wavelength of 13.5 nm, and a bandwidth λ/Δ of 1500 and is focused on the sample surface by an ellipsoidal mirror at a grazing incidence angle of deg.(b) REGINE 3D model detail. The sample is mounted vertically and can be scanned with a 3-axis motorized stage. The X and Y position of the sample is controlled with an interferometer to ensure nanometric scanning accuracy. The reflected beam is collected by an EUV cCD detector (Princeton Instruments MTE3, Trenton, New Jersey, United States) with a 2048× 2048 pixel array.Both the sample and the CcD are mounted on co-axial rotation stages to change the grazing incidence of the sample illumination between 0 and 26 deg. </div>


## Disclosures 

The authors declare no conflicts of interest.

## Code and Data Availability 

time but may be obtained from the authors upon reasonable request.



## Acknowledgments 

The authors acknowledge financial support from the European Union's Horizon 2020 research and innovation program under the Marie Sktodowska–Curie Grant Agreement No. 884104 (PSI–FELLOW—II–3i) and from the SAMSUNG Global Research Outreach (GRO) program.

## References 

1. K. S. Yi et al., "Validation of high-throughput AFM for copper pad metrology in high volume manufacturing," Proc. SPIE 12955, 129552U (2024).
2.B.T.Altintas et al., "High-throughput in-lineSEMmetrology forCu padnanotopography for hybrid bonding applications,"in IEEE10th Electron. Syst.-Integr. Technol. Conf. (ESTC), pp.1–6 (2024).
3.J.Rodenburg and A.Maiden, tychography, pp.819–904,Springer International Publishing,Cham (2019).
4. J.R. Fienup, "Phase retrieval algorithms: a comparison," Appl. Opt. 21, 2758–2769 (1982).
5.A.M. Maiden and J. M. Rodenburg, "An improved ptychographical phase retrieval algorithm for diffractive imaging," Ultramicroscopy 109(10), 1256–1262 (2009).
6.B. Wang et al.,"High-fidelity ptychographic imaging of highly periodic structures enabled by vortex high harmonic beams,," Optica 10, 1245–1252 (2023).
7.H. K. Niazi et al., "Critical dimension scatterometry as a scalable solution for hybrid bonding pad recess metrology," in IEEE 73rd Electron. Comp. and Technol. Conf. (ECTC), pp. 1403–1409 (2023).
8.H. Kasai et al., "In-line SEM evaluation technique for Cu pad nanotopography for hybrid bonding applications," in IEEE CPMT Symp. Jpn. (ICSJ), pp. 92–95 (2024).
9.G. Stoilov and T. Dragostinov, "Phase-stepping interferometry: five-frame algorithm with an arbitrary step,"
Opt. Lasers Eng. 28(1), 61–69 (1997).
10.N. Kumaretal.,"Reconstructionof sub-wavelength features andnano-positioningof gratings using coherent Fourier scatterometry,,  Opt. Express 22, 24678–24688 (2014).
11. P. Ansuinelli and I. Mochi,"Tailoring the support constraint of phase retrieval algorithms improves lensless EUV nanostructures metrology,  Proc. SPIE 13215, 1321516 (2024).
12. P. Ansuinelli et al., "Phase retrieval of periodic patterns," Proc. SPIE 13426, 1342608 (2025).
13. H. Lee et al., "Lens-free reflectivetopography for high-resolution wafer inspection,," Sci. Rep. 14(1), 10519(2024).
14.J.M. Rodenburgand H.M.L.Faulkner, "Aphaseretrieval algorithm for shifting illumination," Appl. Phys.
Lett. 85, 4795–4797 (2004).
15. B. Henke, E. Gullikson, and J. Davis, "X-ray interactions: photoabsorption,scattering,transmission, and reflection at e = 50–30.000 ev,z=1–92," At. Data Nucl. Data Tables 54(2),181-342 (1993).
16. Y. Shao et al.,"Wavelength-multiplexed multi-mode EUV reflection ptychography based on automatic differentiation,"  Light: Sci. Appl. 13(1), 196 (2024).
17. C.Gu et al., "Enhanced EUV maskimaging using Fourier ptychographic microscopy," Proc. SPIE13424,
134240S (2025).
18. X. Xu et al., "Phase retrieval of the full vectorial field applied to coherent Fourier scatterometry,"
Opt. Express 25, 29574–29586 (2017).
19. M. Odstrčil et al.,"Towards optimized illumination for high-resolution ptychography," Opt. Express 27,
14981–14997 (2019).
20. M. van de Kerkhof et al., "Diffuser concepts for in-situ wavefront measurements of EUV projection optics,"
Proc. SPIE 10583, 105830S (2018).
21.Y. Esashietal.,"Tabletopextremeultravioletreflectometerforquantitativenanoscalereflectometry, scatterometry, and imaging," Rev. Sci. Instrum. 94, 123705 (2023).
22.W.Coeneetal.,"EUVimaginofnanostructures withoutlenses,"Proc.SPIE13115,13115072024).
23..F.Gt.Huletfinff-axis apertured illumination,," Opt. Express 20, 19050–19059 (2012).
24. A.de Beurset al.,"aPIE: an angle calibration algorithm for reflection ptychography,"Opt. Lett. 47,25.T.e."lmrciofxrmtioludftg 1949–1952 (2022).
mask blank,"J.Micro/Nanopatterning,Mater.Metrol.23(4),041402(2024).


N:Inti primarily working on lensless imaging.



Hojun eedhycbacor'dreomaUityi213,iasr's degree in2015,and hisPhDi221.Hehas benan eie at Samunemiconductr R&D center since 2021.



Wookrae Kim has been a technical leader at Samsung Electronics' Semiconductor R&D Center since 2020, overseeing the development of cutting-edge metrology and inspection (MI) solutions for the advanced semiconductor manufacturing process.His expertise ncompasses optics and its diverse applications, with a focus on imaging and spectroscopic methods for semiconductor MI.His research focuses on soft X-ray metrology, computational imaging, and the characterization of thermal and electrical properties of semiconductor materials.



Junho Shin received his bachelor's degree in mechanical engineering and electrical engineering in 2013, his master's degree in mechanical engineering in 2015, and his PhD in mechanical engineering in 2020 from the Korea Advanced Institute of Science and Technology. He worked as a researcher at the Korea Atomic Energy Research Institute in 2020 and joined Samsung Electronics as an Engineer in 2021.



Yasin Ekinci is the head of the Laboratory for X-ray Nanoscience and Technologies at Paul Scherrer Institute. He obtained his PhD at the Max-Planck Institute in Göttingen, Germany.He worked on various topics of nanoscience and technology, including surface science, EUV lithography, resist materials, lensless imaging, plasmonics, semiconductor nanostructures, biosensors, and nanofluidics. He is an author/co-author of more than 300 publications, including papers, book chapters, and patent applications. He is a fellow of SPIE.

Iacopo Mochi is anoptical physicist.He started working on EUV mask inspection in2008 atthe Center for X-Ray Optics, where he contributed to the design and development of the SHARP microscope. Later, he joined IMEC as an R&D engineer studying SRAF solutions to mitigate EUV mask thre-dimensional effects.In2016,he joined thePaul Scherrer Institute, where heis leading the advanced lthography and metrology group and working on inteerence lithography and lensless imaging for semiconductor applications.

