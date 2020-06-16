// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef ATTRIBUTE_COMBINATION_H
#define ATTRIBUTE_COMBINATION_H

#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureCore.h"

class FeatureCombinationCore final {
public:

   struct FeatureCombinationEntry {
      const FeatureCore * m_pFeature;
   };

   size_t m_cItemsPerBitPackDataUnit;
   size_t m_cFeatures;
   size_t m_iInputData;
   unsigned int m_cLogEnterGenerateModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogExitGenerateModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogEnterApplyModelFeatureCombinationUpdateMessages;
   unsigned int m_cLogExitApplyModelFeatureCombinationUpdateMessages;
   FeatureCombinationEntry m_FeatureCombinationEntry[1];

   EBM_INLINE static size_t GetFeatureCombinationCountBytes(const size_t cFeatures) {
      return sizeof(FeatureCombinationCore) - sizeof(FeatureCombinationCore::FeatureCombinationEntry) + sizeof(FeatureCombinationCore::FeatureCombinationEntry) * cFeatures;
   }

   EBM_INLINE void Initialize(const size_t cFeatures, const size_t iFeatureCombination) {
      m_cFeatures = cFeatures;
      m_iInputData = iFeatureCombination;
      m_cLogEnterGenerateModelFeatureCombinationUpdateMessages = 2;
      m_cLogExitGenerateModelFeatureCombinationUpdateMessages = 2;
      m_cLogEnterApplyModelFeatureCombinationUpdateMessages = 2;
      m_cLogExitApplyModelFeatureCombinationUpdateMessages = 2;
   }

   EBM_INLINE static FeatureCombinationCore * Allocate(const size_t cFeatures, const size_t iFeatureCombination) {
      const size_t cBytes = GetFeatureCombinationCountBytes(cFeatures);
      EBM_ASSERT(0 < cBytes);
      FeatureCombinationCore * const pFeatureCombination = static_cast<FeatureCombinationCore *>(malloc(cBytes));
      if(UNLIKELY(nullptr == pFeatureCombination)) {
         return nullptr;
      }
      pFeatureCombination->Initialize(cFeatures, iFeatureCombination);
      return pFeatureCombination;
   }

   EBM_INLINE static void Free(FeatureCombinationCore * const pFeatureCombination) {
      free(pFeatureCombination);
   }

   EBM_INLINE static FeatureCombinationCore ** AllocateFeatureCombinations(const size_t cFeatureCombinations) {
      LOG(TraceLevelInfo, "Entered FeatureCombination::AllocateFeatureCombinations");

      EBM_ASSERT(0 < cFeatureCombinations);
      FeatureCombinationCore ** const apFeatureCombinations = new (std::nothrow) FeatureCombinationCore * [cFeatureCombinations];
      if(LIKELY(nullptr != apFeatureCombinations)) {
         // we need to set this to zero otherwise our destructor will attempt to free garbage memory pointers if we prematurely call the destructor
         EBM_ASSERT(!IsMultiplyError(sizeof(*apFeatureCombinations), cFeatureCombinations)); // if we were able to allocate this, then we should be able to calculate how much memory to zero
         memset(apFeatureCombinations, 0, sizeof(*apFeatureCombinations) * cFeatureCombinations);
      }
      LOG(TraceLevelInfo, "Exited FeatureCombination::AllocateFeatureCombinations");
      return apFeatureCombinations;
   }

   EBM_INLINE static void FreeFeatureCombinations(const size_t cFeatureCombinations, FeatureCombinationCore ** apFeatureCombinations) {
      LOG(TraceLevelInfo, "Entered FeatureCombination::FreeFeatureCombinations");
      if(nullptr != apFeatureCombinations) {
         EBM_ASSERT(0 < cFeatureCombinations);
         for(size_t i = 0; i < cFeatureCombinations; ++i) {
            FeatureCombinationCore::Free(apFeatureCombinations[i]);
         }
         delete[] apFeatureCombinations;
      }
      LOG(TraceLevelInfo, "Exited FeatureCombination::FreeFeatureCombinations");
   }
};
static_assert(std::is_pod<FeatureCombinationCore>::value, "We have an array at the end of this stucture, so we don't want anyone else derriving something and putting data there, and non-POD data is probably undefined as to what the space after gets filled with");

// these need to be declared AFTER the class above since the size of FeatureCombination isn't set until the class has been completely declared, and constexpr needs the size before constexpr
constexpr size_t GetFeatureCombinationCountBytesConst(const size_t cFeatures) {
   return sizeof(FeatureCombinationCore) - sizeof(FeatureCombinationCore::FeatureCombinationEntry) + sizeof(FeatureCombinationCore::FeatureCombinationEntry) * cFeatures;
}
constexpr size_t k_cBytesFeatureCombinationMax = GetFeatureCombinationCountBytesConst(k_cDimensionsMax);

#ifndef NDEBUG
class FeatureCombinationCheck final {
public:
   FeatureCombinationCheck() {
      // we need two separate functions for determining the maximum size of FeatureCombination, so let's check that they match at runtime
      EBM_ASSERT(k_cBytesFeatureCombinationMax == FeatureCombinationCore::GetFeatureCombinationCountBytes(k_cDimensionsMax));
   }
};
static FeatureCombinationCheck DEBUG_FeatureCombinationCheck; // yes, this gets duplicated for each include, but it's just for debug..
#endif // NDEBUG

#endif // ATTRIBUTE_COMBINATION_H
