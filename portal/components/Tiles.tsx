import React from 'react'

interface TileProps {
  icon?: React.ReactNode
  title: string
  children?: React.ReactNode
}

export function Tile({ icon, title, children }: TileProps) {
  return (
    <div className="info-tile">
      {icon && <span className="info-tile-icon" aria-hidden="true">{icon}</span>}
      <span className="info-tile-copy">
        <span className="info-tile-title">{title}</span>
        {children && <span className="info-tile-body">{children}</span>}
      </span>
    </div>
  )
}

interface TileGridProps {
  children: React.ReactNode
  cols?: number
}

export function TileGrid({ children, cols = 2 }: TileGridProps) {
  return (
    <div className="info-tile-grid" style={{ '--tile-columns': cols } as React.CSSProperties}>
      {children}
    </div>
  )
}
