import { motion } from 'motion/react'
import { Fingerprint } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import { useSession } from '../../context/SessionContext'
import { cn } from '../../lib/utils'
import { spring } from '../../lib/motion'
import { NAV_ITEMS, isActive } from './navItems'

function BackendChip() {
  const { backendOnline } = useSession()
  const label = backendOnline === null ? 'Connecting' : backendOnline ? 'Backend online' : 'Backend offline'
  return (
    <span
      className={cn(
        'hidden items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-medium sm:inline-flex',
        backendOnline === null ? 'border-border-strong text-muted-foreground' : backendOnline ? 'border-success/35 bg-success/10 text-success' : 'border-warning/40 bg-warning/10 text-warning',
      )}
      title="Live status from GET /system/health"
    >
      <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-current" />
      {label}
    </span>
  )
}

export function TopBar() {
  const { userId } = useSession()
  const { pathname } = useLocation()

  return (
    <header className="sticky top-0 z-40 border-b border-border/70 bg-background/75 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link to="/" className="flex shrink-0 items-center gap-2.5 rounded-lg" aria-label="NCISN - Campus overview">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-primary/30 bg-primary/10">
            <Fingerprint className="h-4 w-4 text-primary" strokeWidth={1.5} aria-hidden />
          </span>
          <span className="min-w-0 leading-tight">
            <span className="block text-[13px] font-semibold tracking-tight text-foreground">NCISN</span>
            <span className="block whitespace-nowrap text-[11px] text-muted-foreground">Biometric Access Console</span>
          </span>
        </Link>

        <nav aria-label="Main" className="hidden lg:block">
          <ul className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const active = isActive(item.href, pathname)
              return (
                <li key={item.href}>
                  <Link
                    to={item.href}
                    aria-current={active ? 'page' : undefined}
                    className={cn(
                      'relative flex items-center gap-2 whitespace-nowrap rounded-lg px-3 py-2 text-[13px] font-medium transition-colors duration-200',
                      active ? 'text-foreground' : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    {active && <motion.span layoutId="nav-active" transition={spring} className="absolute inset-0 rounded-lg border border-border-strong bg-raised" />}
                    <item.icon className="relative h-4 w-4" strokeWidth={1.5} aria-hidden />
                    <span className="relative xl:hidden">{item.short}</span>
                    <span className="relative hidden xl:inline">{item.title}</span>
                  </Link>
                </li>
              )
            })}
          </ul>
        </nav>

        <div className="flex shrink-0 items-center gap-2">
          <BackendChip />
          <span
            className="block max-w-[9rem] truncate rounded-full lg:hidden xl:block border border-border-strong bg-raised/60 px-2.5 py-1 font-mono text-[11px] text-muted-foreground"
            title="Local demo identity, not a real account"
          >
            {userId}
          </span>
        </div>
      </div>
    </header>
  )
}
