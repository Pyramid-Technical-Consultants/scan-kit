# Software safety classification

| | |
|---|---|
| Document | SK-CLASS |
| Standard | IEC 62304:2006+AMD1:2015 |
| Safety class | A |
| Product | Scan Kit |

Git history is the revision record for this document.

## Decision

Scan Kit software is **Class A**. The manufacturer’s rationale is the intended use below. Class A means the software cannot contribute to a hazardous situation that could result in injury or damage to health, given that intended use.

## Intended use this class depends on

Scan Kit is an engineering tool for reviewing proton pencil-beam scanning sessions, preparing test plans, editing device configuration, and running related calculations. People use it to inspect logs, compare sessions, and prepare inputs for the treatment system.

Treatment-release judgment is outside the specified use of this software. A person does not rely on Scan Kit as the record that a patient may be treated. Clinical release stays with the procedures and systems designated for that decision.

## Behaviors the rationale covers on purpose

These behaviors are in the product. The Class A decision still holds only while the intended use above holds:

- Display of recalculated dose, including comparison with a planning-system dose.
- Upload of a plan to a room controller and running that plan.
- Editing of device configuration used by the delivery system.

If a later use of any of those behaviors makes Scan Kit an input to a treatment-release decision, the classification is revisited before that behavior is ported or changed. The software is not reclassified by silence.

## What this record is not

This document does not assign a safety class to the treatment system, the room controller, or the planning system. It classifies Scan Kit software only.

## Machine QA intent

The calculations that would support AAPM TG-224 pencil-beam machine QA, and the gaps that remain, are recorded in [tg-224.md](../quality/tg-224.md). That note does not change this classification. Scan Kit is not the record that a machine may treat. Offering that use reopens this classification before the behavior is added.
