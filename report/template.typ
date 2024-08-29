// Latex reference for article
#let fnt-Huge = 24.88pt
#let fnt-huge = 20.74pt
#let fnt-Large = 14.4pt
#let fnt-large = 12pt
#let fnt-normalsize = 10pt
#let fnt-small = 9pt
#let fnt-tiny = 5pt

#let make-cover(title, cover-image, 
                subtitle:"", course-name:"", author:""
  ) = page(margin: 0pt, 
  background: image(cover-image, width: 100%), {
  place(top+left, dx: 0cm, dy: 2cm, 
    block(width: 100%, spacing: 1pt, inset: 5%, {
      text(font: "Roboto Slab", weight: "thin", size: 50pt, fill:blue, title)
      v(20pt)
      if (subtitle.len() > 0){ 
        text(font: "Roboto Slab", weight: "light", size:22pt, fill:white, subtitle)
        v(30pt)
      }
      if (course-name.len() > 0){ 
        text(font: "Roboto Slab", weight: "light", size:22pt, fill:white, course-name)
        v(10pt)
      }
      text(font: "Roboto Slab", weight: "thin", size:24pt, fill:white, author)
  }, fill: color.rgb(0,0,0,50%)))
  place(horizon+left, rotate(-90deg, 
    text(font: "Roboto Slab", weight: "light", fill: white)[Delft University of Technology]))
  place(bottom+left, dx: 10mm, [AAA])
})

#let template(doc) = {
  set page(
    paper: "a4", 
    margin: ("x": 12.5%, "y": 10%), 
    number-align: center, 
    header-ascent: 0pt,
    numbering: (..x) => [#x]
  )

  set text(font: "Arial", size: fnt-normalsize)

  show heading.where(level: 1) : it => align(right, {
    if (it.numbering != none) {
      let number = counter(heading).display(it.numbering)
      pagebreak()
      text(font: "Roboto Slab", weight: "thin", size: 96pt, number)
      v(0pt)
    }
    text(font: "Roboto Slab", weight: "light", size: fnt-Huge, it.body)
  })

  show heading.where(level: 2) : it => text(font: "Roboto Slab", weight: "light", size: fnt-Large, it.body) + v(5pt)
  show heading.where(level: 3) : it => text(font: "Roboto Slab", weight: "light", size: fnt-large, it.body) + v(5pt)

  show: set outline(title: none, indent: true, fill: none)
  show outline.entry.where(level: 1) : it => strong(it)

  doc
}

#let front-matter(content) = {
  // Front matter
  set page(number-align: center, numbering: "i")
  set heading(numbering: none)
  counter(page).update(1)
  
  content
}

#let main-matter(content) = {
  // Main matter
  set page(numbering: "1")
  set heading(numbering: "1.1.1")
  counter(heading).update(0)
  counter(page).update(1)

  outline(title: none)
  
  content
}
  
#let aft-matter(content) = { 
  // Aft matter
  set page(numbering: "1")
  counter(heading).update(0)

  heading(numbering: none, level: 1)[References]
  content
}