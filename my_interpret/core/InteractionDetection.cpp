// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <string.h> // memset
#include <stdlib.h> // malloc, realloc, free
#include <stddef.h> // size_t, ptrdiff_t
#include <limits> // numeric_limits

#include "ebmcore.h"
#include "EbmInternal.h"
#include "Logging.h" // EBM_ASSERT & LOG
// feature includes
#include "FeatureCore.h"
// dataset depends on features
#include "DataSetByFeature.h"
// depends on the above
#include "DimensionMultiple.h"

class EbmInteractionState {
public:
   const ptrdiff_t m_runtimeLearningTypeOrCountTargetClasses;

   const size_t m_cFeatures;
   // TODO : in the future, we can allocate this inside a function so that even the objects inside are const
   FeatureCore * const m_aFeatures;
   DataSetByFeature * m_pDataSet;

   unsigned int m_cLogEnterMessages;
   unsigned int m_cLogExitMessages;

   EbmInteractionState(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const size_t cFeatures)
      : m_runtimeLearningTypeOrCountTargetClasses(runtimeLearningTypeOrCountTargetClasses)
      , m_cFeatures(cFeatures)
      , m_aFeatures(0 == cFeatures || IsMultiplyError(sizeof(FeatureCore), cFeatures) ? nullptr : static_cast<FeatureCore *>(malloc(sizeof(FeatureCore) * cFeatures)))
      , m_pDataSet(nullptr)
      , m_cLogEnterMessages (1000)
      , m_cLogExitMessages(1000) {
   }

   ~EbmInteractionState() {
      LOG(TraceLevelInfo, "Entered ~EbmInteractionState");

      delete m_pDataSet;
      free(m_aFeatures);

      LOG(TraceLevelInfo, "Exited ~EbmInteractionState");
   }

   bool InitializeInteraction(const EbmCoreFeature * const aFeatures, const size_t cInstances, const void * const aTargets, const IntegerDataType * const aBinnedData, const FractionalDataType * const aPredictorScores) {
      LOG(TraceLevelInfo, "Entered InitializeInteraction");

      if(0 != m_cFeatures && nullptr == m_aFeatures) {
         LOG(TraceLevelWarning, "WARNING InitializeInteraction 0 != m_cFeatures && nullptr == m_aFeatures");
         return true;
      }

      LOG(TraceLevelInfo, "InitializeInteraction starting feature processing");
      if(0 != m_cFeatures) {
         EBM_ASSERT(!IsMultiplyError(m_cFeatures, sizeof(*aFeatures))); // if this overflows then our caller should not have been able to allocate the array
         const EbmCoreFeature * pFeatureInitialize = aFeatures;
         const EbmCoreFeature * const pFeatureEnd = &aFeatures[m_cFeatures];
         EBM_ASSERT(pFeatureInitialize < pFeatureEnd);
         size_t iFeatureInitialize = 0;
         do {
            static_assert(FeatureTypeCore::OrdinalCore == static_cast<FeatureTypeCore>(FeatureTypeOrdinal), "FeatureTypeCore::OrdinalCore must have the same value as FeatureTypeOrdinal");
            static_assert(FeatureTypeCore::NominalCore == static_cast<FeatureTypeCore>(FeatureTypeNominal), "FeatureTypeCore::NominalCore must have the same value as FeatureTypeNominal");
            EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType || FeatureTypeNominal == pFeatureInitialize->featureType);
            FeatureTypeCore featureTypeCore = static_cast<FeatureTypeCore>(pFeatureInitialize->featureType);

            IntegerDataType countBins = pFeatureInitialize->countBins;
            EBM_ASSERT(0 <= countBins); // we can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)
            if(!IsNumberConvertable<size_t, IntegerDataType>(countBins)) {
               LOG(TraceLevelWarning, "WARNING InitializeInteraction !IsNumberConvertable<size_t, IntegerDataType>(countBins)");
               return true;
            }
            size_t cBins = static_cast<size_t>(countBins);
            if(cBins <= 1) {
               EBM_ASSERT(0 != cBins || 0 == cInstances);
               LOG(TraceLevelInfo, "INFO InitializeInteraction feature with 0/1 value");
            }

            EBM_ASSERT(0 == pFeatureInitialize->hasMissing || 1 == pFeatureInitialize->hasMissing);
            bool bMissing = 0 != pFeatureInitialize->hasMissing;

            // this is an in-place new, so there is no new memory allocated, and we already knew where it was going, so we don't need the resulting pointer returned
            new (&m_aFeatures[iFeatureInitialize]) FeatureCore(cBins, iFeatureInitialize, featureTypeCore, bMissing);
            // we don't allocate memory and our constructor doesn't have errors, so we shouldn't have an error here

            EBM_ASSERT(0 == pFeatureInitialize->hasMissing); // TODO : implement this, then remove this assert
            EBM_ASSERT(FeatureTypeOrdinal == pFeatureInitialize->featureType); // TODO : implement this, then remove this assert

            ++iFeatureInitialize;
            ++pFeatureInitialize;
         } while(pFeatureEnd != pFeatureInitialize);
      }
      LOG(TraceLevelInfo, "InitializeInteraction done feature processing");

      LOG(TraceLevelInfo, "Entered DataSetByFeature");
      EBM_ASSERT(nullptr == m_pDataSet);
      if(0 != cInstances) {
         m_pDataSet = new (std::nothrow) DataSetByFeature(m_cFeatures, m_aFeatures, cInstances, aBinnedData, aTargets, aPredictorScores, m_runtimeLearningTypeOrCountTargetClasses);
         if(nullptr == m_pDataSet || m_pDataSet->IsError()) {
            LOG(TraceLevelWarning, "WARNING InitializeInteraction nullptr == pDataSet || pDataSet->IsError()");
            return true;
         }
      }
      LOG(TraceLevelInfo, "Exited DataSetByFeature");

      LOG(TraceLevelInfo, "Exited InitializeInteraction");
      return false;
   }
};





// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
EbmInteractionState * AllocateCoreInteraction(IntegerDataType countFeatures, const EbmCoreFeature * features, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, IntegerDataType countInstances, const void * targets, const IntegerDataType * binnedData, const FractionalDataType * predictorScores) {
   EBM_ASSERT(0 <= countFeatures);
   EBM_ASSERT(0 == countFeatures || nullptr != features);
   // countTargetClasses is checked by our caller since it's only valid for classification at this point
   EBM_ASSERT(0 <= countInstances);
   EBM_ASSERT(0 == countInstances || nullptr != targets);
   EBM_ASSERT(0 == countInstances || 0 == countFeatures || nullptr != binnedData);
   // predictorScores can be null

   if(!IsNumberConvertable<size_t, IntegerDataType>(countFeatures)) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction !IsNumberConvertable<size_t, IntegerDataType>(countFeatures)");
      return nullptr;
   }
   if(!IsNumberConvertable<size_t, IntegerDataType>(countInstances)) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction !IsNumberConvertable<size_t, IntegerDataType>(countInstances)");
      return nullptr;
   }

   size_t cFeatures = static_cast<size_t>(countFeatures);
   size_t cInstances = static_cast<size_t>(countInstances);

   LOG(TraceLevelInfo, "Entered EbmInteractionState");
   EbmInteractionState * const pEbmInteractionState = new (std::nothrow) EbmInteractionState(runtimeLearningTypeOrCountTargetClasses, cFeatures);
   LOG(TraceLevelInfo, "Exited EbmInteractionState %p", static_cast<void *>(pEbmInteractionState));
   if(UNLIKELY(nullptr == pEbmInteractionState)) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction nullptr == pEbmInteractionState");
      return nullptr;
   }
   if(UNLIKELY(pEbmInteractionState->InitializeInteraction(features, cInstances, targets, binnedData, predictorScores))) {
      LOG(TraceLevelWarning, "WARNING AllocateCoreInteraction pEbmInteractionState->InitializeInteraction");
      delete pEbmInteractionState;
      return nullptr;
   }
   return pEbmInteractionState;
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionRegression(
   IntegerDataType countFeatures,
   const EbmCoreFeature * features,
   IntegerDataType countInstances,
   const FractionalDataType * targets,
   const IntegerDataType * binnedData,
   const FractionalDataType * predictorScores
) {
   LOG(TraceLevelInfo, "Entered InitializeInteractionRegression: countFeatures=%" IntegerDataTypePrintf ", features=%p, countInstances=%" IntegerDataTypePrintf ", targets=%p, binnedData=%p, predictorScores=%p", countFeatures, static_cast<const void *>(features), countInstances, static_cast<const void *>(targets), static_cast<const void *>(binnedData), static_cast<const void *>(predictorScores));
   PEbmInteraction pEbmInteraction = reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(countFeatures, features, k_Regression, countInstances, targets, binnedData, predictorScores));
   LOG(TraceLevelInfo, "Exited InitializeInteractionRegression %p", static_cast<void *>(pEbmInteraction));
   return pEbmInteraction;
}

EBMCORE_IMPORT_EXPORT PEbmInteraction EBMCORE_CALLING_CONVENTION InitializeInteractionClassification(
   IntegerDataType countFeatures,
   const EbmCoreFeature * features,
   IntegerDataType countTargetClasses,
   IntegerDataType countInstances,
   const IntegerDataType * targets,
   const IntegerDataType * binnedData,
   const FractionalDataType * predictorScores
) {
   LOG(TraceLevelInfo, "Entered InitializeInteractionClassification: countFeatures=%" IntegerDataTypePrintf ", features=%p, countTargetClasses=%" IntegerDataTypePrintf ", countInstances=%" IntegerDataTypePrintf ", targets=%p, binnedData=%p, predictorScores=%p", countFeatures, static_cast<const void *>(features), countTargetClasses, countInstances, static_cast<const void *>(targets), static_cast<const void *>(binnedData), static_cast<const void *>(predictorScores));
   if(countTargetClasses < 0) {
      LOG(TraceLevelError, "ERROR InitializeInteractionClassification countTargetClasses can't be negative");
      return nullptr;
   }
   if(0 == countTargetClasses && 0 != countInstances) {
      LOG(TraceLevelError, "ERROR InitializeInteractionClassification countTargetClasses can't be zero unless there are no instances");
      return nullptr;
   }
   if(!IsNumberConvertable<ptrdiff_t, IntegerDataType>(countTargetClasses)) {
      LOG(TraceLevelWarning, "WARNING InitializeInteractionClassification !IsNumberConvertable<ptrdiff_t, IntegerDataType>(countTargetClasses)");
      return nullptr;
   }
   const ptrdiff_t runtimeLearningTypeOrCountTargetClasses = static_cast<ptrdiff_t>(countTargetClasses);
   PEbmInteraction pEbmInteraction = reinterpret_cast<PEbmInteraction>(AllocateCoreInteraction(countFeatures, features, runtimeLearningTypeOrCountTargetClasses, countInstances, targets, binnedData, predictorScores));
   LOG(TraceLevelInfo, "Exited InitializeInteractionClassification %p", static_cast<void *>(pEbmInteraction));
   return pEbmInteraction;
}

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static IntegerDataType GetInteractionScorePerTargetClasses(EbmInteractionState * const pEbmInteractionState, const FeatureCombinationCore * const pFeatureCombination, FractionalDataType * const pInteractionScoreReturn) {
   // TODO : be smarter about our CachedInteractionThreadResources, otherwise why have it?
   CachedInteractionThreadResources * const pCachedThreadResources = new (std::nothrow) CachedInteractionThreadResources();

   if(CalculateInteractionScore<compilerLearningTypeOrCountTargetClasses, 0>(pEbmInteractionState->m_runtimeLearningTypeOrCountTargetClasses, pCachedThreadResources, pEbmInteractionState->m_pDataSet, pFeatureCombination, pInteractionScoreReturn)) {
      delete pCachedThreadResources;
      return 1;
   }
   delete pCachedThreadResources;
   return 0;
}

template<ptrdiff_t possibleCompilerLearningTypeOrCountTargetClasses>
EBM_INLINE IntegerDataType CompilerRecursiveGetInteractionScore(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, EbmInteractionState * const pEbmInteractionState, const FeatureCombinationCore * const pFeatureCombination, FractionalDataType * const pInteractionScoreReturn) {
   static_assert(IsClassification(possibleCompilerLearningTypeOrCountTargetClasses), "possibleCompilerLearningTypeOrCountTargetClasses needs to be a classification");
   EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
   if(runtimeLearningTypeOrCountTargetClasses == possibleCompilerLearningTypeOrCountTargetClasses) {
      EBM_ASSERT(runtimeLearningTypeOrCountTargetClasses <= k_cCompilerOptimizedTargetClassesMax);
      return GetInteractionScorePerTargetClasses<possibleCompilerLearningTypeOrCountTargetClasses>(pEbmInteractionState, pFeatureCombination, pInteractionScoreReturn);
   } else {
      return CompilerRecursiveGetInteractionScore<possibleCompilerLearningTypeOrCountTargetClasses + 1>(runtimeLearningTypeOrCountTargetClasses, pEbmInteractionState, pFeatureCombination, pInteractionScoreReturn);
   }
}

template<>
EBM_INLINE IntegerDataType CompilerRecursiveGetInteractionScore<k_cCompilerOptimizedTargetClassesMax + 1>(const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, EbmInteractionState * const pEbmInteractionState, const FeatureCombinationCore * const pFeatureCombination, FractionalDataType * const pInteractionScoreReturn) {
   UNUSED(runtimeLearningTypeOrCountTargetClasses);
   // it is logically possible, but uninteresting to have a classification with 1 target class, so let our runtime system handle those unlikley and uninteresting cases
   static_assert(IsClassification(k_cCompilerOptimizedTargetClassesMax), "k_cCompilerOptimizedTargetClassesMax needs to be a classification");
   EBM_ASSERT(IsClassification(runtimeLearningTypeOrCountTargetClasses));
   EBM_ASSERT(k_cCompilerOptimizedTargetClassesMax < runtimeLearningTypeOrCountTargetClasses);
   return GetInteractionScorePerTargetClasses<k_DynamicClassification>(pEbmInteractionState, pFeatureCombination, pInteractionScoreReturn);
}

// we made this a global because if we had put this variable inside the EbmInteractionState object, then we would need to dereference that before getting the count.  By making this global we can send a log message incase a bad EbmInteractionState object is sent into us
// we only decrease the count if the count is non-zero, so at worst if there is a race condition then we'll output this log message more times than desired, but we can live with that
static unsigned int g_cLogGetInteractionScoreParametersMessages = 10;

EBMCORE_IMPORT_EXPORT IntegerDataType EBMCORE_CALLING_CONVENTION GetInteractionScore(
   PEbmInteraction ebmInteraction,
   IntegerDataType countFeaturesInCombination,
   const IntegerDataType * featureIndexes,
   FractionalDataType * interactionScoreReturn
) {
   LOG_COUNTED(&g_cLogGetInteractionScoreParametersMessages, TraceLevelInfo, TraceLevelVerbose, "GetInteractionScore parameters: ebmInteraction=%p, countFeaturesInCombination=%" IntegerDataTypePrintf ", featureIndexes=%p, interactionScoreReturn=%p", static_cast<void *>(ebmInteraction), countFeaturesInCombination, static_cast<const void *>(featureIndexes), static_cast<void *>(interactionScoreReturn));

   EBM_ASSERT(nullptr != ebmInteraction);
   EbmInteractionState * pEbmInteractionState = reinterpret_cast<EbmInteractionState *>(ebmInteraction);

   LOG_COUNTED(&pEbmInteractionState->m_cLogEnterMessages, TraceLevelInfo, TraceLevelVerbose, "Entered GetInteractionScore");

   EBM_ASSERT(0 <= countFeaturesInCombination);
   EBM_ASSERT(0 == countFeaturesInCombination || nullptr != featureIndexes);
   // interactionScoreReturn can be nullptr

   if(!IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)) {
      LOG(TraceLevelWarning, "WARNING GetInteractionScore !IsNumberConvertable<size_t, IntegerDataType>(countFeaturesInCombination)");
      return 1;
   }
   size_t cFeaturesInCombination = static_cast<size_t>(countFeaturesInCombination);
   if(0 == cFeaturesInCombination) {
      LOG(TraceLevelInfo, "INFO GetInteractionScore empty feature combination");
      if(nullptr != interactionScoreReturn) {
         *interactionScoreReturn = 0; // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our caler be smarter about this condition
      }
      return 0;
   }

   if(nullptr == pEbmInteractionState->m_pDataSet) {
      // if pEbmInteractionState->m_pDataSet is null, then we have a dataset with zero instances.  If there are zero data cases, there isn't much basis to say whether there are interactions, so just return zero
      LOG(TraceLevelInfo, "INFO GetInteractionScore zero instances");
      if(nullptr != interactionScoreReturn) {
         *interactionScoreReturn = 0; // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our caler be smarter about this condition
      }
      return 0;
   }

   const FeatureCore * const aFeatures = pEbmInteractionState->m_aFeatures;
   const IntegerDataType * pFeatureCombinationIndex = featureIndexes;
   const IntegerDataType * const pFeatureCombinationIndexEnd = featureIndexes + cFeaturesInCombination;

   do {
      const IntegerDataType indexFeatureInterop = *pFeatureCombinationIndex;
      EBM_ASSERT(0 <= indexFeatureInterop);
      if(!IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop)) {
         LOG(TraceLevelWarning, "WARNING GetInteractionScore !IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop)");
         return 1;
      }
      size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
      EBM_ASSERT(iFeatureForCombination < pEbmInteractionState->m_cFeatures);
      const FeatureCore * const pFeature = &aFeatures[iFeatureForCombination];
      if(pFeature->m_cBins <= 1) {
         LOG(TraceLevelInfo, "INFO GetInteractionScore feature with 0/1 value");
         if(nullptr != interactionScoreReturn) {
            *interactionScoreReturn = 0; // we return the lowest value possible for the interaction score, but we don't return an error since we handle it even though we'd prefer our caler be smarter about this condition
         }
         return 0;
      }
      ++pFeatureCombinationIndex;
   } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndex);

   if(k_cDimensionsMax < cFeaturesInCombination) {
      // if we try to run with more than k_cDimensionsMax we'll exceed our memory capacity, so let's exit here instead
      LOG(TraceLevelWarning, "WARNING GetInteractionScore k_cDimensionsMax < cFeaturesInCombination");
      return 1;
   }

   // put the pFeatureCombination object on the stack. We want to put it into a FeatureCombination object since we want to share code with training, which calls things like building the tensor totals (which is templated to be compiled many times)
   char FeatureCombinationBuffer[k_cBytesFeatureCombinationMax];
   FeatureCombinationCore * const pFeatureCombination = reinterpret_cast<FeatureCombinationCore *>(&FeatureCombinationBuffer);
   pFeatureCombination->Initialize(cFeaturesInCombination, 0);

   pFeatureCombinationIndex = featureIndexes; // restart from the start
   FeatureCombinationCore::FeatureCombinationEntry * pFeatureCombinationEntry = &pFeatureCombination->m_FeatureCombinationEntry[0];
   do {
      const IntegerDataType indexFeatureInterop = *pFeatureCombinationIndex;
      EBM_ASSERT(0 <= indexFeatureInterop);
      EBM_ASSERT((IsNumberConvertable<size_t, IntegerDataType>(indexFeatureInterop))); // we already checked indexFeatureInterop was good above
      size_t iFeatureForCombination = static_cast<size_t>(indexFeatureInterop);
      EBM_ASSERT(iFeatureForCombination < pEbmInteractionState->m_cFeatures);
      const FeatureCore * const pFeature = &aFeatures[iFeatureForCombination];
      EBM_ASSERT(2 <= pFeature->m_cBins); // we should have filtered out anything with 1 bin above

      pFeatureCombinationEntry->m_pFeature = pFeature;
      ++pFeatureCombinationEntry;
      ++pFeatureCombinationIndex;
   } while(pFeatureCombinationIndexEnd != pFeatureCombinationIndex);

   IntegerDataType ret;
   if(IsRegression(pEbmInteractionState->m_runtimeLearningTypeOrCountTargetClasses)) {
      ret = GetInteractionScorePerTargetClasses<k_Regression>(pEbmInteractionState, pFeatureCombination, interactionScoreReturn);
   } else {
      EBM_ASSERT(IsClassification(pEbmInteractionState->m_runtimeLearningTypeOrCountTargetClasses));
      if(pEbmInteractionState->m_runtimeLearningTypeOrCountTargetClasses <= ptrdiff_t { 1 }) {
         LOG(TraceLevelInfo, "INFO GetInteractionScore target with 0/1 classes");
         if(nullptr != interactionScoreReturn) {
            *interactionScoreReturn = 0; // if there is only 1 classification target, then we can predict the outcome with 100% accuracy and there is no need for logits or interactions or anything else.  We return 0 since interactions have no benefit
         }
         return 0;
      }
      ret = CompilerRecursiveGetInteractionScore<2>(pEbmInteractionState->m_runtimeLearningTypeOrCountTargetClasses, pEbmInteractionState, pFeatureCombination, interactionScoreReturn);
   }
   if(0 != ret) {
      LOG(TraceLevelWarning, "WARNING GetInteractionScore returned %" IntegerDataTypePrintf, ret);
   }
   if(nullptr != interactionScoreReturn) {
      EBM_ASSERT(0 <= *interactionScoreReturn);
      LOG_COUNTED(&pEbmInteractionState->m_cLogExitMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GetInteractionScore %" FractionalDataTypePrintf, *interactionScoreReturn);
   } else {
      LOG_COUNTED(&pEbmInteractionState->m_cLogExitMessages, TraceLevelInfo, TraceLevelVerbose, "Exited GetInteractionScore");
   }
   return ret;
}

EBMCORE_IMPORT_EXPORT void EBMCORE_CALLING_CONVENTION FreeInteraction(
   PEbmInteraction ebmInteraction
) {
   LOG(TraceLevelInfo, "Entered FreeInteraction: ebmInteraction=%p", static_cast<void *>(ebmInteraction));
   EbmInteractionState * pEbmInteractionState = reinterpret_cast<EbmInteractionState *>(ebmInteraction);
   EBM_ASSERT(nullptr != pEbmInteractionState);
   delete pEbmInteractionState;
   LOG(TraceLevelInfo, "Exited FreeInteraction");
}
