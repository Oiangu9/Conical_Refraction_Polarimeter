# Development from Scratch of a State of the Art Polarimeter based on Conical Refraction
---
---
### Project Abstract for the Bachelor's Thesis in Computational Mathematics and Data Analytics
#### *Universitat Autònoma de Barcelona (UAB)*
---

**Student presenting the project :**

Xabier Oyanguren Asua.

**Project Supervisor :**

Àlex Turpin Avilés, La Caixa Junior Leader Principal Investigator at ICFO, Assistant Professsor at UAB (*Grup d’Òptica, Departament de Física*).

#### Project Abstract :

As amazing as it may sound, the Conical Refraction (CR) phenomenon mathematically discovered by Hamilton back in 1832 [1] and experimentally verified by Lloyd a year later, which was actually one of the principal phenomena that led to the imposition of the wave-like description of light in front of the particle-like description defended by scientists like Newton, was left forgotten in the history of science, except for some exceptional works by prominent scientists like Poggendorff (1839) [2], Raman (1941) [3] [4], Belsky and Khapalyuk (1978) [5] [6] and Berry (2004) [7]. These studies were nevertheless mainly theoretical in motivation, and it was not squeezed experimentally until in the last decades, with the Optics Group at UAB leading the initiative, that it regained interest in the scientific literature [8]-[12]. It turns out that among other interesting suggested applications, as reviewed in [15], this phenomenon allows the construction of a theoretically very resolutive polarimeter as it was suggested and explained in References [13] and [14]. It is possible to build a device to measure the variation of an input linearly polarized light's polarization, that needs no mechanical polarization filter manipulation, which would ideally allow a measurement resolution proportional to the resolution of the imaging cameras commercially available.

The objective of the project is to develop the idea of the polarimeter [13] based on Conical Refraction, virtually from scratch, until a fully functional and ideally state of the art portable device. The starting point will be some private fundamental work done by Prof. Todor Kalkandjiev on algorithms to turn the CR ring images to polarization measurements. From there, the student is fully involved both in the algorithm and user interface software design and research, as well as the hardware and housing design for the prototype device. Overall, the next points are being followed by the student, under the supervision of Dr. Turpin:

- Develop additional possible non-black-box algorithms to solve the problem.
\item Implement them together with the algorithms of Prof. Kalkandjiev and build a mock user interface to interact with them.
- Integrate this preliminary user interface in the mock device in the Conical Refraction Lab and allow continous take of measurements and output of results.
- Learn the experimental difficulties found in the laboratory when dealing with the phenomenon and potential optical device-components. Learn how to manipulate them and calculate their required parameters.
- Expand the non-black-box algorithms adapting Deep Learning approaches to the problem. Understand the experimental noise and try to mock it. Train the algorithms.
- Develop mixed approaches between the two classes of algorithms (deep learning and non-black-box algorithms).
- Benchmark the algorithms and as many of their variants as possible with some experimental references. Educe the best found methods to compute the polarization of the input light.
- Create a final user interface for the device, which allows refined measurements and continuous measurements in an intuitive and appealing way.
- Contribute in the design and physical creation of the device prototype, from computational and optical hardware to its housing.
- Integrate the developed software in the final physically realized prototype.
- Benchmark the device with different references and compare it to other polarimeters.
- Based on the results, help find the niches in which the designed polarimeter could compete with currently available devices (by price, by portability, by resolution etc.). 
- Compute the cost-balance, compare it with commercially available polarimeters and educe a final price for the device.
- Build a second prototype as an internal reproducibility-check.

If possible, the thesis project that began in September with the first points of the milestone route, expects to have an ideally marketable prototype that can compete in some niches with currently available polarimeters, by the end of summer.

### References
\[1\]:	W. R. Hamilton, Trans. R. Irish Acad. 17, 1 (1837).

\[2\]:	J. C. Poggendorff, Ann. Phys. Chem. 124, 461 (1839).
	
\[3\]:	C. V. Raman, V. S. Rajagopalan, and T. M. K. Nedungadi, Nature 147, 268 (1941). 
	
\[4\]:	C. V. Raman, Curr. Sci. 11, 46 (1942).

\[5\]:	A. M. Belsky and A. P. Khapalyuk, Opt. Spectrosc. 44, 436 (1978).
	
\[6\]:	A. M. Belsky and A. P. Khapalyuk, Opt. Spectrosc. 44, 312 (1978).
	
\[7\]:	M. V. Berry, J. Opt. A: Pure Appl. Opt. 6, 289 (2004).
	
\[8\]:	T. K. Kalkandjiev and M. A. Bursukova, Proc. SPIE 6994, 69940B (2008).

\[9\]:	S. D. Grant, S. A. Zolotovskaya, T. K. Kalkandjiev, W. A. Gillespie, and A. Abdolvand, Opt. Express 22, 21347 (2014).

\[10\]:	A. Peinado, A. Turpin, C. Iemmi, A. Márquez, T. K. Kalkandjiev, J. Mompart, and J. Campos, Opt. Express 23, 18080 (2015).
	
\[11\]:	A. Turpin, J. Polo, Y. V. Loiko, J. KAber, F. Schmaltz, T. K. Kalkandjiev, V. Ahufinger, G. Birkl, and J. Mompart, Opt. Express 23, 1638 (2015).	

\[12\]:	A. Turpin, Yu. V. Loiko, T. K. Kalkandjiev, R. Corbalán, and J. Mompart, Phys. Rev. A 92, 013802 (2015).

\[13\]:	Lizana, A.; Estévez, I.; Turpin, A.; Ramirez, C.; Peinado, A.; Campos, J. (2015). {\em "Implementation and performance of an in-line incomplete Stokes polarimeter based on a single biaxial crystal"}. Applied Optics, 54(29), 8758–. doi:10.1364/AO.54.008758 

\[14\]:	Grant S D, Reynolds S and Abdolvand A 2016 {\em "Optical sensing of polarization using conical diffraction phenomenon"} J. Opt. 18 025609
	
\[15\]:	Turpin, A., Loiko, Yu. V., Kalkandjiev, T. K. and Mompart, J. {\em "Conical Refraction: fundamentals and applications"}. Laser Photon. Rev. 10, 750–771 (2016).
	
    
