import React from 'react'

interface FeatureProps {
  icon: React.ReactNode
  title: string
  children: React.ReactNode
}

export function Feature({ icon, title, children }: FeatureProps) {
  return (
    <div className="portal-feature">
      <span className="portal-feature-icon" aria-hidden="true">{icon}</span>
      <span>
        <strong>{title}</strong>
        <span>: {children}</span>
      </span>
    </div>
  )
}

export function Features({ children }: { children: React.ReactNode }) {
  return <div className="portal-features">{children}</div>
}