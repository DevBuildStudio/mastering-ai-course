import React from 'react'

interface IconListItemProps {
  icon: React.ReactNode
  children: React.ReactNode
}

export function IconListItem({ icon, children }: IconListItemProps) {
  return (
    <li className="icon-list-item">
      <span className="icon-list-icon" aria-hidden="true">{icon}</span>
      <span>{children}</span>
    </li>
  )
}

export function IconList({ children }: { children: React.ReactNode }) {
  return <ul className="icon-list">{children}</ul>
}
