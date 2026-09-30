import { Activity, KeyRound, LayoutGrid, LockKeyhole, ShieldCheck, type LucideIcon } from 'lucide-react'

export interface NavItem {
  title: string
  /** Label for the compact mobile tab bar. */
  short: string
  icon: LucideIcon
  href: string
}

// The console's sections (unchanged routes). "Campus" also covers the building flow (/building/...).
export const NAV_ITEMS: NavItem[] = [
  { title: 'Campus', short: 'Campus', icon: LayoutGrid, href: '/' },
  { title: 'Model Testing', short: 'Testing', icon: Activity, href: '/testing' },
  { title: 'Security Analytics', short: 'Analytics', icon: ShieldCheck, href: '/analytics' },
  { title: 'Template Management', short: 'Templates', icon: KeyRound, href: '/templates' },
  { title: 'Template Protection', short: 'Protection', icon: LockKeyhole, href: '/template-protection' },
]

export function isActive(href: string, pathname: string): boolean {
  return href === '/' ? pathname === '/' || pathname.startsWith('/building/') : pathname.startsWith(href)
}
