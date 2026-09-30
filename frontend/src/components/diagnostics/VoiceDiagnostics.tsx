import type { VoiceQualityCheckResponse } from '../../api/types'
import { DiagnosticsBox, type DiagnosticRow } from './DiagnosticsBox'

const n = (v: number | null | undefined, digits = 1, unit = '') => (v === null || v === undefined ? '-' : `${v.toFixed(digits)}${unit}`)

/** What the voice quality gate measured on ONE recording and which check decided (development only). */
export function VoiceDiagnostics({ result, fileBytes }: { result: VoiceQualityCheckResponse; fileBytes?: number }) {
  const d = result.diagnostics
  if (!d) return null
  const m = result.metrics
  const snr = m.estimated_snr_db ?? null
  const speech = m.speech_seconds ?? null
  const clipping = m.clipping_fraction ?? null
  const rows: DiagnosticRow[] = [
    { label: 'Recording (as uploaded)', value: `${n(d.input_duration_seconds, 2, ' s')} - ${d.input_sample_rate_hz ?? '-'} Hz - ${d.channels ?? '-'} ch${fileBytes ? ` - ${(fileBytes / 1024).toFixed(0)} KB` : ''}` },
    { label: 'Level: RMS / peak', value: `${n(d.rms_dbfs, 1)} / ${n(d.peak_dbfs, 1)} dBFS` },
    { label: 'Speech level / noise floor', value: `${n(d.speech_level_dbfs, 1)} / ${n(d.noise_floor_dbfs, 1)} dBFS` },
    { label: 'Speech / silence', value: `${n(speech, 2)} s / ${n(d.silence_seconds, 2)} s` },
    {
      label: `Speech >= ${d.limits.min_speech_seconds} s`,
      value: n(speech, 2, ' s'),
      status: speech !== null && speech >= d.limits.min_speech_seconds ? 'pass' : 'fail',
    },
    {
      label: `Clipping <= ${(d.limits.max_clipping_fraction * 100).toFixed(1)} %`,
      value: n(clipping !== null ? clipping * 100 : null, 2, ' %'),
      status: clipping !== null && clipping <= d.limits.max_clipping_fraction ? 'pass' : 'fail',
    },
    {
      label: `Estimated SNR >= ${d.limits.min_estimated_snr_db} dB`,
      value: snr === null ? 'no noise floor' : n(snr, 1, ' dB'),
      status: snr === null || snr >= d.limits.min_estimated_snr_db ? 'pass' : 'fail',
    },
    { label: 'Speech-band SNR (100-4000 Hz)', value: n(d.speech_band_snr_db, 1, ' dB') },
    { label: 'Energy below 100 Hz (hum)', value: n(d.low_frequency_energy_ratio !== null && d.low_frequency_energy_ratio !== undefined ? d.low_frequency_energy_ratio * 100 : null, 1, ' %') },
  ]
  return (
    <DiagnosticsBox
      title="Voice capture diagnostics"
      className="mb-3"
      rows={rows}
      footer={
        <>
          Decision: <span className={result.passed ? 'text-success' : 'text-danger'}>{result.passed ? 'PASS' : 'FAIL'}</span> ({result.verdict})
          {d.failed_check && (
            <>
              {' '}- failing check: <span className="font-mono text-foreground">{d.failed_check.metric}</span> = {d.failed_check.measured ?? '-'} (required{' '}
              {d.failed_check.required})
            </>
          )}
        </>
      }
    />
  )
}
