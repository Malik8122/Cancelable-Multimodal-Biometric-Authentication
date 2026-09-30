import { AnimatePresence, MotionConfig } from 'motion/react'
import { useState } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import { BackendStatusBanner } from './components/layout/BackendStatusBanner'
import { NavDock } from './components/layout/NavDock'
import { PageTransition, type NavDirection } from './components/layout/PageTransition'
import { SystemAmbient } from './components/layout/SystemAmbient'
import { TopBar } from './components/layout/TopBar'
import { AnalyticsPage } from './pages/AnalyticsPage'
import { AuthenticatePage } from './pages/AuthenticatePage'
import { CheckpointPage } from './pages/CheckpointPage'
import { LandingPage } from './pages/LandingPage'
import { RegisterPage } from './pages/RegisterPage'
import { ResultPage } from './pages/ResultPage'
import { TemplateManagementPage } from './pages/TemplateManagementPage'
import { TemplateProtectionPage } from './pages/TemplateProtectionPage'
import { TestingPage } from './pages/TestingPage'

// Depth of a route along the authentication flow; null = a section outside the flow.
function flowDepth(pathname: string): number | null {
  if (pathname === '/') return 0
  if (/^\/building\/[^/]+\/?$/.test(pathname)) return 1
  if (/^\/building\/[^/]+\/(register|authenticate)\/?$/.test(pathname)) return 2
  if (/^\/building\/[^/]+\/result\/?$/.test(pathname)) return 3
  return null
}

function directionBetween(from: string, to: string): NavDirection {
  const a = flowDepth(from)
  const b = flowDepth(to)
  if (a === null || b === null || a === b) return 0
  return b > a ? 1 : -1
}

function App() {
  const location = useLocation()
  // Derived during render from the previous path (React's "adjust state on prop change" pattern), so the entering and
  // leaving pages agree on the direction.
  const [nav, setNav] = useState<{ path: string; direction: NavDirection }>({ path: location.pathname, direction: 0 })
  if (nav.path !== location.pathname) setNav({ path: location.pathname, direction: directionBetween(nav.path, location.pathname) })

  return (
    // reducedMotion="user": people who ask their OS for reduced motion get no transform/layout animation anywhere.
    <MotionConfig reducedMotion="user">
    <div className="relative flex min-h-screen flex-col text-foreground">
      <SystemAmbient />
      <a
        href="#main"
        className="sr-only z-50 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
      >
        Skip to content
      </a>
      <TopBar />
      <BackendStatusBanner />
      <main id="main" className="relative flex-1 pb-24 lg:pb-12">
        <AnimatePresence mode="wait" initial={false} custom={nav.direction}>
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<PageTransition><LandingPage /></PageTransition>} />
            <Route path="/building/:buildingId" element={<PageTransition><CheckpointPage /></PageTransition>} />
            <Route path="/building/:buildingId/register" element={<PageTransition><RegisterPage /></PageTransition>} />
            <Route
              path="/building/:buildingId/authenticate"
              element={<PageTransition><AuthenticatePage /></PageTransition>}
            />
            <Route path="/building/:buildingId/result" element={<PageTransition><ResultPage /></PageTransition>} />
            <Route path="/testing" element={<PageTransition><TestingPage /></PageTransition>} />
            <Route path="/analytics" element={<PageTransition><AnalyticsPage /></PageTransition>} />
            <Route path="/templates" element={<PageTransition><TemplateManagementPage /></PageTransition>} />
            <Route path="/template-protection" element={<PageTransition><TemplateProtectionPage /></PageTransition>} />
          </Routes>
        </AnimatePresence>
      </main>
      <NavDock />
    </div>
    </MotionConfig>
  )
}

export default App
