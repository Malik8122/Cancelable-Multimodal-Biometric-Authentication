// Biometric protocol tests (node:test, run with `npm test`). ENROLLMENT and AUTHENTICATION are separate protocols:
//   registration  1 Face (5 samples) -> 2 Voice (N enrollment sentences) -> 3 Dynamic Hand Gesture (3 independent Z samples)
//   authentication  Face 1 capture, Voice 1 sentence, Hand 1 Z gesture -> fusion -> decision
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, it } from 'node:test'
import {
  FACE_AUTHENTICATION_CAPTURES,
  HAND_AUTHENTICATION_SAMPLES,
  HAND_ENROLLMENT_SAMPLES,
  REGISTRATION_STEPS,
  registrationStep,
  registrationStepLabel,
  sampleCounter,
  VOICE_AUTHENTICATION_PROMPT,
  VOICE_AUTHENTICATION_SENTENCES,
  VOICE_ENROLLMENT_PROMPTS,
  handSamples,
  voicePrompts,
} from '../src/config/protocol.ts'
import { buildFusionSamples } from '../src/lib/authSamples.ts'

const source = (path: string) => readFileSync(new URL(`../src/${path}`, import.meta.url), 'utf8')
const blob = (text: string) => new Blob([text])

describe('registration order', () => {
  it('numbers the biometric steps 1 Face, 2 Voice, 3 Dynamic Hand Gesture', () => {
    assert.deepEqual(
      REGISTRATION_STEPS.map((s) => [s.step, s.modality]),
      [
        [1, 'face'],
        [2, 'voice'],
        [3, 'hand'],
      ],
    )
    assert.equal(registrationStep('face').step, 1)
    assert.equal(registrationStep('voice').step, 2)
    assert.equal(registrationStep('hand').step, 3)
  })

  it('labels the cards from the protocol, with no fourth step', () => {
    assert.equal(registrationStepLabel('face'), 'Step 1')
    assert.equal(registrationStepLabel('voice'), 'Step 2')
    assert.equal(registrationStepLabel('hand'), 'Step 3 - Optional')
    const page = source('pages/RegisterPage.tsx')
    assert.doesNotMatch(page, /Step 4/)
    assert.doesNotMatch(page, /step="Step \d/, 'step labels must come from registrationStepLabel(), not hard-coded strings')
  })
})

describe('voice protocol', () => {
  it('enrollment records every configured enrollment prompt (more than one)', () => {
    assert.ok(VOICE_ENROLLMENT_PROMPTS.length > 1)
    assert.deepEqual(voicePrompts('register'), VOICE_ENROLLMENT_PROMPTS)
  })

  it('authentication asks for two sentences', () => {
    assert.equal(VOICE_AUTHENTICATION_SENTENCES, 2)
    assert.equal(voicePrompts('verify').length, 2)
    assert.equal(voicePrompts('verify')[0], VOICE_AUTHENTICATION_PROMPT)
  })

  it('both voice sentences are sent at authentication, the second as voice_audio_2', () => {
    const samples = buildFusionSamples(['voice'], {
      voice: [
        { blob: blob('sentence one'), filename: 'voice-sentence-1.wav' },
        { blob: blob('sentence two'), filename: 'voice-sentence-2.wav' },
      ],
    })
    assert.deepEqual(samples.map((s) => s.filename), ['voice-sentence-1.wav', 'voice-sentence-2.wav'])
    assert.match(source('api/client.ts'), /voice_audio_2/)
  })
})

describe('hand gesture protocol', () => {
  it('enrollment takes three independent samples, authentication one', () => {
    assert.equal(HAND_ENROLLMENT_SAMPLES, 3)
    assert.equal(HAND_AUTHENTICATION_SAMPLES, 1)
    assert.equal(handSamples('register'), 3)
    assert.equal(handSamples('verify'), 1)
  })

  it('shows the sample counter as 01 / 03 ... 03 / 03', () => {
    assert.deepEqual(
      [1, 2, 3].map((n) => sampleCounter(n, HAND_ENROLLMENT_SAMPLES)),
      ['01 / 03', '02 / 03', '03 / 03'],
    )
  })

  it('enrollment appends each accepted sample and never describes repeating one gesture several times', () => {
    const enrollment = source('components/capture/HandGestureEnrollment.tsx')
    assert.match(enrollment, /\[\.\.\.accepted, blob\]/, 'each accepted sample is appended, never replacing an earlier one')
    for (const file of ['pages/RegisterPage.tsx', 'components/capture/HandGestureEnrollment.tsx', 'components/capture/HandGestureCapture.tsx']) {
      assert.doesNotMatch(source(file), /(three|five|3|5) times/i, `${file} describes repeating one gesture`)
    }
  })

  it('the gesture is the Z - the infinity gesture is gone', () => {
    assert.match(source('config/hand.ts'), /GESTURE_TYPE = 'z'/)
    assert.match(source('hand/landmarker.ts'), /mirrored: true/, 'the payload says the preview was mirrored')
    for (const file of ['config/hand.ts', 'config/protocol.ts', 'pages/RegisterPage.tsx', 'components/capture/HandGestureCapture.tsx',
      'components/capture/HandGestureEnrollment.tsx', 'components/biometric/FusionReveal.tsx']) {
      // (the JavaScript `Infinity` keyword, e.g. an animation's `repeat: Infinity`, is not the gesture)
      assert.doesNotMatch(source(file), /'infinity'|infinity (gesture|sign|shape)|\u221e|∞|lemniscate/i, `${file} still mentions the infinity gesture`)
    }
  })

  it('verification shows no enrollment counter', () => {
    const auth = source('pages/AuthenticatePage.tsx')
    assert.doesNotMatch(auth, /HAND_ENROLLMENT_SAMPLES|sampleCounter/, 'authentication must not show "1 / 3"')
  })
})

describe('authentication request', () => {
  it('sends exactly one sample per presented factor, in order', () => {
    const captured = {
      face: { blob: blob('face'), filename: 'face.png' },
      voice: { blob: blob('voice'), filename: 'voice.wav' },
      hand: { blob: blob('hand'), filename: 'gesture.json' },
    }
    const samples = buildFusionSamples(['face', 'voice', 'hand'], captured)
    assert.deepEqual(
      samples.map((s) => [s.modality, s.filename]),
      [
        ['face', 'face.png'],
        ['voice', 'voice.wav'],
        ['hand', 'gesture.json'],
      ],
    )
    assert.equal(FACE_AUTHENTICATION_CAPTURES, 1)
  })

  it('refuses to build a request for a factor that was not captured (never a silent pass)', () => {
    assert.throws(() => buildFusionSamples(['face', 'voice'], { face: { blob: blob('f'), filename: 'f.png' } }), /voice has not been captured/)
  })
})

describe('enrollment card actions', () => {
  it('an enrolled modality has exactly one action, "Re-enroll" - no duplicate "Update Enrollment"', () => {
    const page = source('pages/RegisterPage.tsx')
    assert.doesNotMatch(page, /Update Enrollment/, 'the duplicate "Update Enrollment" action is gone')
    // the button label sits on its own line; the help sentence "use <span>Re-enroll</span> on its card" is not a button
    assert.equal(page.match(/^\s*Re-enroll\s*$/gm)?.length ?? 0, 1, 'exactly one Re-enroll button label (shared by every modality card)')
    assert.match(page, /Enroll \$\{MODALITY_LABEL\[modality\]\}/, 'a not-yet-enrolled modality keeps its Enroll action')
  })
})
