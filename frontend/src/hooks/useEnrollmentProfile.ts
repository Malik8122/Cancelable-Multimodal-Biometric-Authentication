import { useCallback, useEffect, useState } from 'react'
import { getEnrollmentStatus } from '../api/client'
import { ApiError, type EnrollmentStatus, type Modality } from '../api/types'
import { APPLICATION_ID } from '../config/app'

/**
 * The user's enrollment profile: which modalities they have enrolled and the status of each
 * (NOT_REGISTERED / REGISTERED / UPDATED / RETRY_REQUIRED). It is the user - not the building - who decides what to
 * enroll and what to present, so nothing here depends on a facility.
 */
export function useEnrollmentProfile(userId: string) {
  const [enrolled, setEnrolled] = useState<Partial<Record<Modality, boolean>> | null>(null)
  const [statuses, setStatuses] = useState<Partial<Record<Modality, EnrollmentStatus>>>({})
  const [displayName, setDisplayName] = useState<string | null>(null)
  const [hasDisplayName, setHasDisplayName] = useState(false)
  const [profileEditingEnabled, setProfileEditingEnabled] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const profile = await getEnrollmentStatus(userId, APPLICATION_ID)
      setEnrolled(profile.modalities)
      setStatuses(profile.statuses)
      setDisplayName(profile.display_name)
      setHasDisplayName(profile.has_display_name)
      setProfileEditingEnabled(!!profile.profile_editing_enabled)
      setError(null)
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'The backend is unreachable.')
    } finally {
      setLoading(false)
    }
  }, [userId])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const enrolledList = (Object.keys(enrolled ?? {}) as Modality[]).filter((m) => enrolled?.[m])
  return { enrolled, statuses, enrolledList, displayName, hasDisplayName, profileEditingEnabled, error, loading, refresh }
}
