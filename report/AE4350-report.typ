#import "template.typ": *

#show: doc => template(doc)

// \newfontfamily\titlestyle{Roboto Slab Light}
// \newfontfamily\largetitlestyle{Roboto Slab Thin}

#make-cover(
  "A Title to the Report", //"ADHDP-optimized acceleration-based raceing",
  "cover.jpg",
  course-name: "AE4350",
  author: "Nick Felten"
)

#front-matter()[
= Abstract

]

#main-matter()[

= Introduction
The 

= Methodology
The 
// Explain adhdp

= Results
The 

= Conclusion
The 

@adhdp_flow
@og_adhdp
@sutton-barto
@barto-1995

]


#aft-matter()[

#bibliography("bibliography.bib", title: none)

]