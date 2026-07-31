import React from 'react'
import Link from 'next/link'

interface CardProps {
  icon?: React.ReactNode
  eyebrow?: string
  title: string
  href: string
  children?: React.ReactNode
}

export function Card({ icon, eyebrow, title, href, children }: CardProps) {
  return (
    <Link href={href} className="course-card">
      <span className="course-card-icon">{icon}</span>
      <span className="course-card-copy">
        {eyebrow && <span className="course-card-eyebrow">{eyebrow}</span>}
        <span className="course-card-title">{title}</span>
        {children && <span className="course-card-description">{children}</span>}
      </span>
      <span className="course-card-arrow" aria-hidden="true">→</span>
    </Link>
  )
}

interface CardsProps {
  children: React.ReactNode
  cols?: number
}

export function Cards({ children, cols = 2 }: CardsProps) {
  return (
    <div className="course-card-grid" style={{ '--card-columns': cols } as React.CSSProperties}>
      {children}
    </div>
  )
}