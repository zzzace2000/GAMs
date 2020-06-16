// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef INITIALIZE_RESIDUALS_H
#define INITIALIZE_RESIDUALS_H

#include <stddef.h> // size_t, ptrdiff_t

#include "ebmcore.h"
#include "EbmStatistics.h"
#include "Logging.h" // EBM_ASSERT & LOG

// a*PredictorScores = logOdds for binary classification
// a*PredictorScores = logWeights for multiclass classification
// a*PredictorScores = predictedValue for regression
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
static void InitializeResiduals(const size_t cInstances, const void * const aTargetData, const FractionalDataType * const aPredictorScores, FractionalDataType * pResidualError, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG(TraceLevelInfo, "Entered InitializeResiduals");

   // TODO : review this function to see if iZeroResidual was set to a valid index, does that affect the number of items in pPredictorScores (I assume so), and does it affect any calculations below like sumExp += std::exp(predictionScore) and the equivalent.  Should we use cVectorLength or runtimeLearningTypeOrCountTargetClasses for some of the addition
   // TODO : !!! re-examine the idea of zeroing one of the residuals with iZeroResidual.  Do we get exact equivalent results if we initialize them the correct way.  Try debugging this by first doing a binary as multiclass (2 == cVectorLength) and seeing if our algorithm is re-startable (do 2 cycles and then try doing 1 cycle and exiting then re-creating it with aPredictionScore values and doing a 2nd cycle and see if it gives the same results).  It would be a huge win to be able to consitently eliminate one residual value!).  Maybe try construcing a super-simple dataset with 10 cases and 1 feature and see how it behaves
   EBM_ASSERT(0 < cInstances);
   EBM_ASSERT(nullptr != aTargetData);
   EBM_ASSERT(nullptr != pResidualError);

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   EBM_ASSERT(0 < cVectorLength);
   EBM_ASSERT(!IsMultiplyError(cVectorLength, cInstances)); // if we couldn't multiply these then we should not have been able to allocate pResidualError before calling this function
   const size_t cVectoredItems = cVectorLength * cInstances;
   EBM_ASSERT(!IsMultiplyError(cVectoredItems, sizeof(pResidualError[0]))); // if we couldn't multiply these then we should not have been able to allocate pResidualError before calling this function
   const FractionalDataType * const pResidualErrorEnd = pResidualError + cVectoredItems;

   if(nullptr == aPredictorScores) {
      // TODO: do we really need to handle the case where pPredictorScores is null? In the future, we'll probably initialize our data with the intercept, in which case we'll always have existing predictions
      if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
         // calling ComputeRegressionResidualError(predictionScore, data) with predictionScore as zero gives just data, so we can memcopy these values
         memcpy(pResidualError, aTargetData, cInstances * sizeof(pResidualError[0]));
#ifndef NDEBUG
         const FractionalDataType * pTargetData = static_cast<const FractionalDataType *>(aTargetData);
         do {
            const FractionalDataType data = *pTargetData;
            EBM_ASSERT(!std::isnan(data));
            EBM_ASSERT(!std::isinf(data));
            const FractionalDataType predictionScore = 0;
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(predictionScore, data);
            EBM_ASSERT(*pResidualError == residualError);
            ++pTargetData;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
#endif // NDEBUG
      } else {
         EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));

         const IntegerDataType * pTargetData = static_cast<const IntegerDataType *>(aTargetData);

         const FractionalDataType matchValue = EbmStatistics::ComputeClassificationResidualErrorMulticlass(true, static_cast<FractionalDataType>(cVectorLength));
         const FractionalDataType nonMatchValue = EbmStatistics::ComputeClassificationResidualErrorMulticlass(false, static_cast<FractionalDataType>(cVectorLength));

         EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
         const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);

         do {
            const IntegerDataType targetOriginal = *pTargetData;
            EBM_ASSERT(0 <= targetOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(targetOriginal))); // if we can't fit it, then we should increase our StorageDataTypeCore size!
            const StorageDataTypeCore target = static_cast<StorageDataTypeCore>(targetOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, ptrdiff_t>(runtimeLearningTypeOrCountTargetClasses)));
            EBM_ASSERT(target < static_cast<StorageDataTypeCore>(runtimeLearningTypeOrCountTargetClasses));

            if(IsBinaryClassification(compilerLearningTypeOrCountTargetClasses)) {
               const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorBinaryclass(target);
               *pResidualError = residualError;
               ++pResidualError;
            } else {
               for(StorageDataTypeCore iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorMulticlass(target, iVector, matchValue, nonMatchValue);
                  EBM_ASSERT(EbmStatistics::ComputeClassificationResidualErrorMulticlass(static_cast<FractionalDataType>(cVectorLength), 0, target, iVector) == residualError);
                  *pResidualError = residualError;
                  ++pResidualError;
               }
               // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
               // 
               // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
               // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
               // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of values
               // insted of allowing them to be scaled.  
               // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which will be multiplication outside the exp(..), which
               // means the numerator and denominator are multiplied by the same constant, which cancels eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
               constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
               if(bZeroingResiduals) {
                  pResidualError[k_iZeroResidual - static_cast<ptrdiff_t>(cVectorLength)] = 0;
               }
            }
            ++pTargetData;
         } while(pResidualErrorEnd != pResidualError);
      }
   } else {
      const FractionalDataType * pPredictorScores = aPredictorScores;
      if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
         const FractionalDataType * pTargetData = static_cast<const FractionalDataType *>(aTargetData);
         do {
            const FractionalDataType data = *pTargetData;
            EBM_ASSERT(!std::isnan(data));
            EBM_ASSERT(!std::isinf(data));
            const FractionalDataType predictionScore = *pPredictorScores;
            const FractionalDataType residualError = EbmStatistics::ComputeRegressionResidualError(predictionScore, data);
            *pResidualError = residualError;
            ++pTargetData;
            ++pPredictorScores;
            ++pResidualError;
         } while(pResidualErrorEnd != pResidualError);
      } else {
         EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));

         const IntegerDataType * pTargetData = static_cast<const IntegerDataType *>(aTargetData);

         EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, size_t>(cVectorLength)));
         const StorageDataTypeCore cVectorLengthStorage = static_cast<StorageDataTypeCore>(cVectorLength);

         do {
            const IntegerDataType targetOriginal = *pTargetData;
            EBM_ASSERT(0 <= targetOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, IntegerDataType>(targetOriginal))); // if we can't fit it, then we should increase our StorageDataTypeCore size!
            const StorageDataTypeCore target = static_cast<StorageDataTypeCore>(targetOriginal);
            EBM_ASSERT((IsNumberConvertable<StorageDataTypeCore, ptrdiff_t>(runtimeLearningTypeOrCountTargetClasses)));
            EBM_ASSERT(target < static_cast<StorageDataTypeCore>(runtimeLearningTypeOrCountTargetClasses));
            if(IsBinaryClassification(compilerLearningTypeOrCountTargetClasses)) {
               const FractionalDataType predictionScore = *pPredictorScores;
               const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorBinaryclass(predictionScore, target);
               *pResidualError = residualError;
               ++pPredictorScores;
               ++pResidualError;
            } else {
               FractionalDataType sumExp = 0;
               // TODO : eventually eliminate this subtract variable once we've decided how to handle removing one logit
               const FractionalDataType subtract = 0 <= k_iZeroClassificationLogitAtInitialize ? pPredictorScores[k_iZeroClassificationLogitAtInitialize] : 0;

               for(StorageDataTypeCore iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FractionalDataType predictionScore = *pPredictorScores - subtract;
                  sumExp += std::exp(predictionScore);
                  ++pPredictorScores;
               }

               // go back to the start so that we can iterate again
               pPredictorScores -= cVectorLengthStorage;

               for(StorageDataTypeCore iVector = 0; iVector < cVectorLengthStorage; ++iVector) {
                  const FractionalDataType predictionScore = *pPredictorScores - subtract;
                  // TODO : we're calculating exp(predictionScore) above, and then again in ComputeClassificationResidualErrorMulticlass.  exp(..) is expensive so we should just do it once instead and store the result in a small memory array here
                  const FractionalDataType residualError = EbmStatistics::ComputeClassificationResidualErrorMulticlass(sumExp, predictionScore, target, iVector);
                  *pResidualError = residualError;
                  ++pPredictorScores;
                  ++pResidualError;
               }
               // TODO: this works as a way to remove one parameter, but it obviously insn't as efficient as omitting the parameter
               // 
               // this works out in the math as making the first model vector parameter equal to zero, which in turn removes one degree of freedom
               // from the model vector parameters.  Since the model vector weights need to be normalized to sum to a probabilty of 100%, we can set the first
               // one to the constant 1 (0 in log space) and force the other parameters to adjust to that scale which fixes them to a single valid set of values
               // insted of allowing them to be scaled.  
               // Probability = exp(T1 + I1) / [exp(T1 + I1) + exp(T2 + I2) + exp(T3 + I3)] => we can add a constant inside each exp(..) term, which will be multiplication outside the exp(..), which
               // means the numerator and denominator are multiplied by the same constant, which cancels eachother out.  We can thus set exp(T2 + I2) to exp(0) and adjust the other terms
               constexpr bool bZeroingResiduals = 0 <= k_iZeroResidual;
               if(bZeroingResiduals) {
                  pResidualError[k_iZeroResidual - static_cast<ptrdiff_t>(cVectorLengthStorage)] = 0;
               }
            }
            ++pTargetData;
         } while(pResidualErrorEnd != pResidualError);
      }
   }
   LOG(TraceLevelInfo, "Exited InitializeResiduals");
}

#endif // INITIALIZE_RESIDUALS_H
