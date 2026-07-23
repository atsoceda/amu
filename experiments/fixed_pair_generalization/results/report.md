# Fixed-Pair Generalization Report

Generated: 2026-07-23T09:39:01.232740+00:00
Model: `google/gemma-3-270m`

## Question

Does the fixed `L13/F10304 + L14/F1949` suppression discovered on `ophthalmologist` improve article preparation on other occupations without corrupting consonant controls?

## Design

- Demonstration: `Someone who studies living organisms is a biologist.`
- Held-out occupation prompts: 107
- Expected `an` prompts: 21
- Expected `a` controls: 86
- The ophthalmologist source sentence was excluded.
- The same two feature identities were suppressed at each prompt's final token. No held-out feature selection was performed.

## Short Answer

The fixed pair changed held-out behavior, but it did not generalize as a clean content-preserving preparation intervention. It repaired some expected-`an` labels while also changing many expected-`a` controls and frequently changing the answer word itself.

## Outcome Counts

| Outcome | Count |
| --- | ---: |
| Expected-`an` prompts corrected to generated `an` | 10 |
| Expected-`an` prompts regressed away from generated `an` | 0 |
| Expected-`a` controls incorrectly changed to generated `an` | 22 |
| Expected-`a` controls preserved as generated `a` | 64 |
| All prompts whose top token changed | 33 |
| All prompts whose generated article changed | 31 |
| Continuations whose first content word changed | 37 |
| Realized article/word agreement repaired | 0 |
| Realized article/word agreement regressed | 0 |
| Exact listed-word completions before intervention | 53 |
| Exact listed-word completions after intervention | 44 |

## Prompts Whose Top Article Changed

| Prompt | Expected | Baseline | Intervention | Baseline Continuation | Intervention Continuation | Content Changed? | Δ(`an-a`) |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `Someone who flies airplanes is` | `a` | `a` | `an` | ` a pilot.` | ` an aviator.` | True | 0.875 |
| `Someone who takes professional pictures is` | `a` | `a` | `an` | ` a photographer.` | ` an artist.` | True | 1.125 |
| `Someone who designs buildings and structures is` | `an` | `a` | `an` | ` a designer.` | ` an architect.` | True | 1.375 |
| `Someone who represents clients in legal matters is` | `a` | `a` | `an` | ` a lawyer.` | ` an attorney.` | True | 1.000 |
| `Someone who studies matter and energy is` | `a` | `a` | `an` | ` a physicist.` | ` an astronomer.` | True | 1.000 |
| `Someone who studies chemical reactions and compounds is` | `a` | `a` | `an` | ` a chemist.` | ` an organic chemist.` | True | 1.250 |
| `Someone who studies human behavior and mental processes is` | `a` | `a` | `an` | ` a psychologist.` | ` an anthropologist.` | True | 1.375 |
| `Someone who studies financial systems and markets is` | `an` | `a` | `an` | ` a financial analyst.` | ` an economist.` | True | 0.875 |
| `Someone who studies stars and planets is` | `an` | `a` | `an` | ` a physicist.` | ` an astronomer.` | True | 1.000 |
| `Someone who studies rocks and earth formations is` | `a` | `a` | `an` | ` a geologist.` | ` an archaeologist.` | True | 1.250 |
| `Someone who works with numbers and mathematical theories is` | `a` | `a` | `an` | ` a mathematician.` | ` an economist.` | True | 1.000 |
| `Someone who analyzes data and calculates probabilities is` | `a` | `a` | `an` | ` a statistician.` | ` an economist.` | True | 1.125 |
| `Someone who studies animals is` | `a` | `a` | `an` | ` a zoologist.` | ` an ecologist.` | True | 1.125 |
| `Someone who studies ancient civilizations through artifacts is` | `an` | `a` | `an` | ` a historian.` | ` an archaeologist.` | True | 1.000 |
| `Someone who performs in plays or movies is` | `an` | `a` | `an` | ` a director.` | ` an actor.` | True | 0.875 |
| `Someone who creates written works is` | `a` | `a` | `an` | ` a scientist.` | ` an anthropologist.` | True | 1.250 |
| `Someone who makes or repairs jewelry is` | `a` | `a` | `an` | ` a jeweler.` | ` an artist.` | True | 1.375 |
| `Someone who applies paint to surfaces is` | `a` | `a` | `an` | ` a painter.` | ` an artist.` | True | 0.875 |
| `Someone who tends to plants and landscapes is` | `a` | `a` | `an` | ` a botanist.` | ` an ecologist.` | True | 1.125 |
| `Someone who grows crops and raises livestock is` | `a` | `a` | `an` | ` a farmer.` | ` an agriculturist.` | True | 1.250 |
| `Someone who provides expert advice is` | `a` | `a` | `an` | ` a psychologist.` | ` an economist.` | True | 1.000 |
| `Someone who examines data and information is` | `an` | `a` | `an` | ` a statistician.` | ` an economist.` | True | 1.250 |
| `Someone who examines financial records is` | `an` | `a` | `an` | ` a financial analyst.` | ` an accountant.` | True | 1.000 |
| `Someone who works in financial institutions is` | `a` | `a` | `an` | ` a financial analyst.` | ` an economist.` | True | 0.750 |
| `Someone who helps buy and sell property is` | `a` | `a` | `an` | ` a real estate agent` | ` an economist.` | True | 1.000 |
| `Someone who sells products or services is` | `a` | `a` | `an` | ` a marketer.` | ` an economist.` | True | 1.000 |
| `Someone who sells goods directly to consumers is` | `a` | `a` | `an` | ` a marketer.` | ` an economist.` | True | 1.125 |
| `Someone who sells goods in large quantities is` | `a` | `a` | `an` | ` a marketer.` | ` an economist.` | True | 1.125 |
| `Someone who creates new devices or processes is` | `an` | `a` | `an` | ` a scientist.` | ` an engineer.` | True | 0.875 |
| `Someone who examines things for quality or compliance is` | `an` | `a` | `an` | ` a sociologist.` | ` an economist.` | True | 0.750 |
| `Someone who tests or evaluates people or things is` | `an` | `a` | `an` | ` a psychologist.` | ` an anthropologist.` | True | 1.125 |

## Every Held-Out Prompt

| Prompt | Listed Word | Expected | Baseline Generated Article | Intervention Generated Article | Δ(`an-a`) | Baseline Continuation | Intervention Continuation |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `Someone who heals sick pets is` | `veterinarian` | `a` | `a` | `a` | 1.125 | ` a physician.` | ` a physician.` |
| `Someone who educates children in schools is` | `teacher` | `a` | `a` | `a` | 1.125 | ` a teacher.` | ` a teacher.` |
| `Someone who prepares meals in restaurants is` | `chef` | `a` | `a` | `a` | 1.000 | ` a chef.` | ` a chef.` |
| `Someone who extinguishes fires and rescues people is` | `firefighter` | `a` | `a` | `a` | 0.750 | ` a firefighter.` | ` a firefighter.` |
| `Someone who flies airplanes is` | `pilot` | `a` | `a` | `an` | 0.875 | ` a pilot.` | ` an aviator.` |
| `Someone who treats teeth and gums is` | `dentist` | `a` | `a` | `a` | 1.125 | ` a dentist.` | ` a dentist.` |
| `Someone who installs and repairs electrical systems is` | `electrician` | `an` | `a` | `a` | 0.875 | ` a technician.` | ` a technician.` |
| `Someone who manages books and helps people find information is` | `librarian` | `a` | `a` | `a` | 1.125 | ` a librarian.` | ` a librarian.` |
| `Someone who takes professional pictures is` | `photographer` | `a` | `a` | `an` | 1.125 | ` a photographer.` | ` an artist.` |
| `Someone who designs buildings and structures is` | `architect` | `an` | `a` | `an` | 1.375 | ` a designer.` | ` an architect.` |
| `Someone who fixes cars and engines is` | `mechanic` | `a` | `a` | `a` | 0.875 | ` a mechanic.` | ` a mechanic.` |
| `Someone who performs operations on patients is` | `surgeon` | `a` | `a` | `a` | 1.000 | ` a surgeon.` | ` a surgeon.` |
| `Someone who dispenses medications is` | `pharmacist` | `a` | `a` | `a` | 1.250 | ` a pharmacist.` | ` a pharmacist.` |
| `Someone who writes news articles is` | `journalist` | `a` | `a` | `a` | 1.125 | ` a journalist.` | ` a journalist.` |
| `Someone who fixes pipes and water systems is` | `plumber` | `a` | `a` | `a` | 1.000 | ` a plumber.` | ` a plumber.` |
| `Someone who manages financial records is` | `accountant` | `an` | `a` | `a` | 1.000 | ` a financial analyst.` | ` a financial analyst.` |
| `Someone who represents clients in legal matters is` | `lawyer` | `a` | `a` | `an` | 1.000 | ` a lawyer.` | ` an attorney.` |
| `Someone who designs and builds technical systems is` | `engineer` | `an` | `a` | `a` | 0.875 | ` a computer scientist.` | ` a computer scientist.` |
| `Someone who makes bread and pastries is` | `baker` | `a` | `a` | `a` | 1.250 | ` a baker.` | ` a baker.` |
| `Someone who builds things with wood is` | `carpenter` | `a` | `a` | `a` | 1.250 | ` a carpenter.` | ` a carpenter.` |
| `Someone who teaches and conducts research at universities is` | `professor` | `a` | `a` | `a` | 1.125 | ` a scientist.` | ` a scientist.` |
| `Someone who studies matter and energy is` | `physicist` | `a` | `a` | `an` | 1.000 | ` a physicist.` | ` an astronomer.` |
| `Someone who studies living organisms is` | `biologist` | `a` | `a` | `a` | 1.125 | ` a biologist.` | ` a biologist.` |
| `Someone who studies chemical reactions and compounds is` | `chemist` | `a` | `a` | `an` | 1.250 | ` a chemist.` | ` an organic chemist.` |
| `Someone who studies and writes about past events is` | `historian` | `a` | `a` | `a` | 1.125 | ` a historian.` | ` a historian.` |
| `Someone who studies human behavior and mental processes is` | `psychologist` | `a` | `a` | `an` | 1.375 | ` a psychologist.` | ` an anthropologist.` |
| `Someone who studies financial systems and markets is` | `economist` | `an` | `a` | `an` | 0.875 | ` a financial analyst.` | ` an economist.` |
| `Someone who treats heart diseases is` | `cardiologist` | `a` | `a` | `a` | 1.250 | ` a cardiologist.` | ` a cardiologist.` |
| `Someone who treats disorders of the nervous system is` | `neurologist` | `a` | `a` | `a` | 1.125 | ` a neurologist.` | ` a neurologist.` |
| `Someone who provides medical care to children is` | `pediatrician` | `a` | `a` | `a` | 1.375 | ` a pediatrician.` | ` a pediatrician.` |
| `Someone who treats skin conditions is` | `dermatologist` | `a` | `a` | `a` | 1.375 | ` a dermatologist.` | ` a dermatologist.` |
| `Someone who treats mental health disorders is` | `psychiatrist` | `a` | `a` | `a` | 1.125 | ` a psychiatrist.` | ` a psychiatrist.` |
| `Someone who interprets medical images is` | `radiologist` | `a` | `a` | `a` | 1.500 | ` a doctor.` | ` a doctor.` |
| `Someone who administers anesthesia during surgery is` | `anesthesiologist` | `an` | `a` | `a` | 1.625 | ` a physician.` | ` a physician.` |
| `Someone who treats cancer patients is` | `oncologist` | `an` | `a` | `a` | 1.375 | ` a physician.` | ` a physician.` |
| `Someone who treats bone and joint problems is` | `orthopedist` | `an` | `a` | `a` | 1.125 | ` a physiologist.` | ` a physiologist.` |
| `Someone who studies stars and planets is` | `astronomer` | `an` | `a` | `an` | 1.000 | ` a physicist.` | ` an astronomer.` |
| `Someone who studies rocks and earth formations is` | `geologist` | `a` | `a` | `an` | 1.250 | ` a geologist.` | ` an archaeologist.` |
| `Someone who studies human cultures and societies is` | `anthropologist` | `an` | `a` | `a` | 1.375 | ` a sociologist.` | ` a sociologist.` |
| `Someone who studies social behavior and institutions is` | `sociologist` | `a` | `a` | `a` | 1.125 | ` a sociologist.` | ` a sociologist.` |
| `Someone who works with numbers and mathematical theories is` | `mathematician` | `a` | `a` | `an` | 1.000 | ` a mathematician.` | ` an economist.` |
| `Someone who analyzes data and calculates probabilities is` | `statistician` | `a` | `a` | `an` | 1.125 | ` a statistician.` | ` an economist.` |
| `Someone who studies plants is` | `botanist` | `a` | `a` | `a` | 1.375 | ` a botanist.` | ` a botanist.` |
| `Someone who studies animals is` | `zoologist` | `a` | `a` | `an` | 1.125 | ` a zoologist.` | ` an ecologist.` |
| `Someone who studies weather patterns is` | `meteorologist` | `a` | `a` | `a` | 1.125 | ` a meteorologist.` | ` a meteorologist.` |
| `Someone who studies ancient civilizations through artifacts is` | `archaeologist` | `an` | `a` | `an` | 1.000 | ` a historian.` | ` an archaeologist.` |
| `Someone who studies languages and their structure is` | `linguist` | `a` | `a` | `a` | 1.125 | ` a linguist.` | ` a linguist.` |
| `Someone who plays instruments or composes music is` | `musician` | `a` | `a` | `a` | 1.125 | ` a composer.` | ` a composer.` |
| `Someone who creates visual art is` | `artist` | `an` | `an` | `an` | 1.125 | ` an artist.` | ` an artist.` |
| `Someone who creates three-dimensional art is` | `sculptor` | `a` | `an` | `an` | 1.125 | ` an artist.` | ` an artist.` |
| `Someone who performs choreographed movements is` | `dancer` | `a` | `a` | `a` | 0.875 | ` a dancer.` | ` a dancer.` |
| `Someone who performs in plays or movies is` | `actor` | `an` | `a` | `an` | 0.875 | ` a director.` | ` an actor.` |
| `Someone who creates written works is` | `writer` | `a` | `a` | `an` | 1.250 | ` a scientist.` | ` an anthropologist.` |
| `Someone who writes poetry is` | `poet` | `a` | `a` | `a` | 1.375 | ` a poet.` | ` a poet.` |
| `Someone who writes novels is` | `novelist` | `a` | `a` | `a` | 1.000 | ` a novelist.` | ` a novelist.` |
| `Someone who reviews and revises written content is` | `editor` | `an` | `a` | `a` | 1.000 | ` a writer.` | ` a writer.` |
| `Someone who converts text from one language to another is` | `translator` | `a` | `a` | `a` | 1.000 | ` a linguist.` | ` a linguist.` |
| `Someone who mixes and serves drinks is` | `bartender` | `a` | `a` | `a` | 1.000 | ` a bartender.` | ` a bartender.` |
| `Someone who serves food in restaurants is` | `waiter` | `a` | `a` | `a` | 1.250 | ` a chef.` | ` a chef.` |
| `Someone who cuts and styles hair is` | `barber` | `a` | `a` | `a` | 1.250 | ` a barber.` | ` a barber.` |
| `Someone who makes or alters clothing is` | `tailor` | `a` | `a` | `a` | 1.500 | ` a fashion designer.` | ` a designer.` |
| `Someone who arranges and sells flowers is` | `florist` | `a` | `a` | `a` | 1.125 | ` a florist.` | ` a florist.` |
| `Someone who makes or repairs jewelry is` | `jeweler` | `a` | `a` | `an` | 1.375 | ` a jeweler.` | ` an artist.` |
| `Someone who repairs timepieces is` | `watchmaker` | `a` | `a` | `a` | 0.750 | ` a chronometerist` | ` a chronometerist` |
| `Someone who makes and repairs locks is` | `locksmith` | `a` | `a` | `a` | 1.250 | ` a locksmith.` | ` a locksmith.` |
| `Someone who joins metal pieces using heat is` | `welder` | `a` | `a` | `a` | 0.875 | ` a physicist.` | ` a chemist.` |
| `Someone who works with stone and brick is` | `mason` | `a` | `a` | `a` | 0.875 | ` a mason.` | ` a mason.` |
| `Someone who installs and repairs roofs is` | `roofer` | `a` | `a` | `a` | 1.125 | ` a builder.` | ` a civil engineer.` |
| `Someone who applies paint to surfaces is` | `painter` | `a` | `a` | `an` | 0.875 | ` a painter.` | ` an artist.` |
| `Someone who tends to plants and landscapes is` | `gardener` | `a` | `a` | `an` | 1.125 | ` a botanist.` | ` an ecologist.` |
| `Someone who grows crops and raises livestock is` | `farmer` | `a` | `a` | `an` | 1.250 | ` a farmer.` | ` an agriculturist.` |
| `Someone who catches fish for a living is` | `fisherman` | `a` | `a` | `a` | 1.250 | ` a biologist.` | ` a biologist.` |
| `Someone who works on ships is` | `sailor` | `a` | `a` | `a` | 1.125 | ` a scientist.` | ` a scientist.` |
| `Someone who drives large trucks for cargo transport is` | `trucker` | `a` | `a` | `a` | 1.125 | ` a truck driver.` | ` a truck driver.` |
| `Someone who leads orchestras or manages trains is` | `conductor` | `a` | `a` | `a` | 0.875 | ` a conductor.` | ` a conductor.` |
| `Someone who presides over legal proceedings is` | `judge` | `a` | `a` | `a` | 1.125 | ` a lawyer.` | ` a lawyer.` |
| `Someone who investigates crimes is` | `detective` | `a` | `a` | `a` | 1.125 | ` a criminologist.` | ` a criminologist.` |
| `Someone who provides emergency medical care is` | `paramedic` | `a` | `a` | `a` | 1.375 | ` a physician.` | ` a physician.` |
| `Someone who coordinates emergency services is` | `dispatcher` | `a` | `a` | `a` | 0.750 | ` a paramedic.` | ` a paramedic.` |
| `Someone who provides guidance and support is` | `counselor` | `a` | `a` | `a` | 1.125 | ` a psychologist.` | ` a psychologist.` |
| `Someone who provides treatment for physical or mental conditions is` | `therapist` | `a` | `a` | `a` | 1.125 | ` a psychologist.` | ` a psychiatrist.` |
| `Someone who advises on diet and nutrition is` | `nutritionist` | `a` | `a` | `a` | 1.250 | ` a nutritionist.` | ` a nutritionist.` |
| `Someone who helps others develop skills or fitness is` | `trainer` | `a` | `a` | `a` | 1.000 | ` a trainer.` | ` a trainer.` |
| `Someone who teaches and guides athletes is` | `coach` | `a` | `a` | `a` | 1.250 | ` a coach.` | ` a coach.` |
| `Someone who enforces rules in sports is` | `referee` | `a` | `a` | `a` | 0.875 | ` a coach.` | ` a coach.` |
| `Someone who handles payments in stores is` | `cashier` | `a` | `a` | `a` | 0.875 | ` a cashier.` | ` a financial analyst.` |
| `Someone who performs administrative tasks is` | `clerk` | `a` | `a` | `a` | 0.625 | ` a manager.` | ` a manager.` |
| `Someone who greets visitors and answers phones is` | `receptionist` | `a` | `a` | `a` | 0.750 | ` a doctor.` | ` a doctor.` |
| `Someone who provides administrative support is` | `secretary` | `a` | `a` | `a` | 0.750 | ` a social scientist.` | ` a social scientist.` |
| `Someone who oversees operations and staff is` | `manager` | `a` | `a` | `a` | 0.875 | ` a manager.` | ` a manager.` |
| `Someone who directs and monitors workers is` | `supervisor` | `a` | `a` | `a` | 1.000 | ` a chemist.` | ` a scientist.` |
| `Someone who provides expert advice is` | `consultant` | `a` | `a` | `an` | 1.000 | ` a psychologist.` | ` an economist.` |
| `Someone who examines data and information is` | `analyst` | `an` | `a` | `an` | 1.250 | ` a statistician.` | ` an economist.` |
| `Someone who examines financial records is` | `auditor` | `an` | `a` | `an` | 1.000 | ` a financial analyst.` | ` an accountant.` |
| `Someone who works in financial institutions is` | `banker` | `a` | `a` | `an` | 0.750 | ` a financial analyst.` | ` an economist.` |
| `Someone who facilitates financial transactions is` | `broker` | `a` | `a` | `a` | 0.875 | ` a banker.` | ` a banker.` |
| `Someone who helps buy and sell property is` | `realtor` | `a` | `a` | `an` | 1.000 | ` a real estate agent` | ` an economist.` |
| `Someone who conducts auctions is` | `auctioneer` | `an` | `a` | `a` | 0.750 | ` a stockbroker.` | ` a stockbroker.` |
| `Someone who sells products or services is` | `salesperson` | `a` | `a` | `an` | 1.000 | ` a marketer.` | ` an economist.` |
| `Someone who sells goods directly to consumers is` | `retailer` | `a` | `a` | `an` | 1.125 | ` a marketer.` | ` an economist.` |
| `Someone who sells goods in large quantities is` | `wholesaler` | `a` | `a` | `an` | 1.125 | ` a marketer.` | ` an economist.` |
| `Someone who creates new devices or processes is` | `inventor` | `an` | `a` | `an` | 0.875 | ` a scientist.` | ` an engineer.` |
| `Someone who writes computer code is` | `programmer` | `a` | `a` | `a` | 0.750 | ` a programmer.` | ` a programmer.` |
| `Someone who maintains and repairs technical equipment is` | `technician` | `a` | `a` | `a` | 1.000 | ` a technician.` | ` a technician.` |
| `Someone who runs machinery or equipment is` | `operator` | `an` | `a` | `a` | 0.875 | ` a scientist.` | ` a scientist.` |
| `Someone who examines things for quality or compliance is` | `inspector` | `an` | `a` | `an` | 0.750 | ` a sociologist.` | ` an economist.` |
| `Someone who tests or evaluates people or things is` | `examiner` | `an` | `a` | `an` | 1.125 | ` a psychologist.` | ` an anthropologist.` |

## Interpretation Boundary

A useful content-preserving intervention must repair multiple held-out expected-`an` cases while producing few or no target-label errors on expected-`a` controls and while keeping the intended content stable. A grammatically valid switch such as `a pilot` to `an aviator` is not a grammatical error, but it is evidence that the intervention changes answer selection rather than merely exposing a fixed future plan. This screen does not reselect features per prompt.
