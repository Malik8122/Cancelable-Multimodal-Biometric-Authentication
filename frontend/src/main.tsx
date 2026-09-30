import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import { BuildingsProvider } from './context/BuildingsContext.tsx'
import { SessionProvider } from './context/SessionContext.tsx'
import { AmbientProvider } from './context/AmbientContext.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <SessionProvider>
        <BuildingsProvider>
          <AmbientProvider>
            <App />
          </AmbientProvider>
        </BuildingsProvider>
      </SessionProvider>
    </BrowserRouter>
  </StrictMode>,
)
