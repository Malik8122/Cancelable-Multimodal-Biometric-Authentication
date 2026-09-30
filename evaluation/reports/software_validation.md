# Software validation

Evidence label: SOFTWARE VALIDATION (not biometric performance).

## backend tests

`python -m pytest -q -p no:cacheprovider --junitxml=evaluation\reports\pytest_junit.xml` -> exit 0 (811s)

```
........................................................................ [ 11%]
........................................................................ [ 22%]
........................................................................ [ 33%]
........................................................................ [ 45%]
........................................................................ [ 56%]
........................................................................ [ 67%]
........................................................................ [ 79%]
........................................................................ [ 90%]
............................................................             [100%]
============================== warnings summary ===============================
tests/test_audit_logging.py::test_fusion_writes_an_audit_entry_with_fusion_fields
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

tests/test_audit_logging.py::test_fusion_writes_an_audit_entry_with_fusion_fields
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
636 passed, 2 warnings in 785.15s (0:13:05)
<sys>:0: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

```

## frontend typecheck

`npx.cmd tsc -b` -> exit 0 (27s)

```

```

## frontend lint

`npm.cmd run lint` -> exit 0 (3s)

```
Calling setState synchronously inside an effect starts another render and is usually unnecessary. Derive the value during render, initialize state directly, or update it from the event that caused the change. Use an effect only when synchronizing with an external system.
src/pages/RegisterPage.tsx:167:5: warning react(set-state-in-effect): Calling setState synchronously within an effect can trigger cascading renders help: Effects should synchronize React with external systems. Calling setState synchronously inside an effect starts another render and is usually unnecessary. Derive the value during render, initialize state directly, or update it from the event that caused the change. Use an effect only when synchronizing with an external system.
src/components/ui/file-upload.tsx:39:5: warning eslint(no-unused-expressions): Expected expression to be used help: Consider using this expression or removing it
src/components/ui/background-beams.tsx:106:38: warning react(purity): Cannot call impure function during render help: `Math.random` is an impure function. Calling an impure function can produce unstable results that update unpredictably when the component re-renders
src/components/ui/background-beams.tsx:109:29: warning react(purity): Cannot call impure function during render help: `Math.random` is an impure function. Calling an impure function can produce unstable results that update unpredictably when the component re-renders
src/components/ui/background-beams.tsx:112:26: warning react(purity): Cannot call impure function during render help: `Math.random` is an impure function. Calling an impure function can produce unstable results that update unpredictably when the component re-renders
src/components/capture/VoiceCapture.tsx:61:10: warning react(refs): Cannot access refs during render help: React refs are values that are not needed for rendering. Refs should only be accessed outside of render, such as in event handlers or effects. Accessing a ref value (the `current` property) during render can cause your component not to update as expected
src/components/ui/button.tsx:66:18: warning react(only-export-components): Fast refresh only works when a file only exports components. Use a new file to share constants or functions between components.
src/components/capture/GuidedFaceCapture.tsx:6:14: warning react(only-export-components): Fast refresh only works when a file only exports components. Use a new file to share constants or functions between components.

```

## frontend build

`npm.cmd run build` -> exit 0 (41s)

```

> frontend@0.0.0 build
> tsc -b && vite build

[36mvite v8.2.2 [32mbuilding client environment for production...[36m[39m
transforming...
âœ“ 8504 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                                              0.52 kB â”‚ gzip:   0.34 kB
dist/assets/geist-cyrillic-ext-wght-normal-DjL33-gN.woff2    7.42 kB
dist/assets/geist-vietnamese-wght-normal-6IgcOCM7.woff2      8.00 kB
dist/assets/geist-cyrillic-wght-normal-BEAKL7Jp.woff2       15.08 kB
dist/assets/geist-latin-ext-wght-normal-DC-KSUi6.woff2      16.51 kB
dist/assets/geist-latin-wght-normal-BgDaEnEv.woff2          29.40 kB
dist/assets/index-CJVO0Mx0.css                              99.63 kB â”‚ gzip:  16.26 kB
dist/assets/index-CrjHc04j.js                              571.06 kB â”‚ gzip: 176.65 kB

[32mâœ“ built in 16.38s[39m
[33m[plugin builtin:vite-reporter] 
(!) Some chunks are larger than 500 kB after minification. Consider:
- Using dynamic import() to code-split the application
- Use build.rolldownOptions.output.codeSplitting to improve chunking: https://rolldown.rs/reference/OutputOptions.codeSplitting
- Adjust chunk size limit for this warning via build.chunkSizeWarningLimit.[39m

```
