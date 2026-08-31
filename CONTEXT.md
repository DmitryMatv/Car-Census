# Car-Census Recognition Context

This context covers how Car-Census obtains and reuses vehicle make/model recognition results while protecting identification precision and controlling external API cost.

## Recognition Evidence

**API-confirmed**:
A recognition result returned by TrafficEye and retained with its source and confidence metadata. It is usable evidence, not immutable ground truth.
_Avoid_: Definitive label, ground truth

**Retrieval cache**:
A durable collection of previously observed vehicle images, recognition results, and provenance that can satisfy a later recognition request without invoking TrafficEye when the match is sufficiently safe.
_Avoid_: Classifier, training cache

**Retrieval match**:
A new vehicle image judged sufficiently equivalent to a previously stored image or observation for selected recognition fields to be reused.
_Avoid_: Prediction, classification

**Near-duplicate observation**:
A vehicle image that is sufficiently close to a stored observation for selected identity fields to be reused, without claiming that it is the same physical vehicle or the same class.
_Avoid_: Same car, same model

**Identity fields**:
The stable vehicle description targeted by retrieval: make, model, generation, and, when evidence supports it, variation.
_Avoid_: Image metadata, observation fields

**Observation fields**:
Properties tied to the queried image or API detection, such as color, view, tags, detection box, and detection confidence.
_Avoid_: Vehicle identity

**Ambiguous match**:
A retrieval candidate that fails the conservative similarity/consensus policy or conflicts with other stored evidence; it must fall through to TrafficEye.
_Avoid_: Low-confidence hit

**Label adjudication**:
A later correction or validation recorded as a new version while preserving the original API response and its provenance.
_Avoid_: In-place correction, ground-truth fix

**Shadow retrieval**:
Evaluation mode in which a proposed local match is recorded but TrafficEye is still called, allowing savings and false-match rates to be measured before enforcement.
_Avoid_: Dry-run classification
