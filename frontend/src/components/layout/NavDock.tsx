import { motion } from 'motion/react'
import { Link, useLocation } from 'react-router-dom'
import { cn } from '../../lib/utils'
import { spring } from '../../lib/motion'
import { NAV_ITEMS, isActive } from './navItems'

// Below the large breakpoint the main navigation is a labelled bottom tab bar (the desktop links live in TopBar).
// Every target is at least 44 px tall and carries a visible label, not an icon alone.
export function NavDock() {
  const { pathname } = useLocation()
  return (
    <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-40 border-t border-border/70 bg-background/85 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl lg:hidden">
      <ul className="mx-auto grid max-w-xl grid-cols-5">
        {NAV_ITEMS.map((item) => {
          const active = isActive(item.href, pathname)
          return (
            <li key={item.href}>
              <Link
                to={item.href}
                aria-current={active ? 'page' : undefined}
                className={cn(
                  'relative flex min-h-14 flex-col items-center justify-center gap-1 text-[11px] font-medium transition-colors duration-200',
                  active ? 'text-primary' : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {active && <motion.span layoutId="tab-active" transition={spring} className="absolute top-0 h-0.5 w-8 rounded-full bg-primary" />}
                <item.icon className="h-5 w-5" strokeWidth={1.5} aria-hidden />
                {item.short}
              </Link>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
