// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef TREE_NODE_H
#define TREE_NODE_H

#include <type_traits> // std::is_pod
#include <stddef.h> // size_t, ptrdiff_t

#include "EbmInternal.h" // EBM_INLINE
#include "Logging.h" // EBM_ASSERT & LOG
#include "SegmentedTensor.h"
#include "EbmStatistics.h"
#include "CachedThreadResources.h"
#include "FeatureCore.h"
#include "SamplingWithReplacement.h"
#include "HistogramBucket.h"

template<bool bClassification>
class TreeNode;

template<bool bClassification>
EBM_INLINE size_t GetTreeNodeSizeOverflow(size_t cVectorLength) {
   return IsMultiplyError(sizeof(HistogramBucketVectorEntry<bClassification>), cVectorLength) ? true : IsAddError(sizeof(TreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>), sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength) ? true : false;
}
template<bool bClassification>
EBM_INLINE size_t GetTreeNodeSize(size_t cVectorLength) {
   return sizeof(TreeNode<bClassification>) - sizeof(HistogramBucketVectorEntry<bClassification>) + sizeof(HistogramBucketVectorEntry<bClassification>) * cVectorLength;
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * AddBytesTreeNode(TreeNode<bClassification> * pTreeNode, size_t countBytesAdd) {
   return reinterpret_cast<TreeNode<bClassification> *>(reinterpret_cast<char *>(pTreeNode) + countBytesAdd);
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * GetLeftTreeNodeChild(TreeNode<bClassification> * pTreeNodeChildren, size_t countBytesTreeNode) {
   UNUSED(countBytesTreeNode);
   return pTreeNodeChildren;
}
template<bool bClassification>
EBM_INLINE TreeNode<bClassification> * GetRightTreeNodeChild(TreeNode<bClassification> * pTreeNodeChildren, size_t countBytesTreeNode) {
   return AddBytesTreeNode<bClassification>(pTreeNodeChildren, countBytesTreeNode);
}

template<bool bClassification>
class TreeNodeData;

template<>
class TreeNodeData<true> {
   // classification version of the TreeNodeData

public:

   struct BeforeExaminationForPossibleSplitting {
      const HistogramBucket<true> * pHistogramBucketEntryFirst;
      const HistogramBucket<true> * pHistogramBucketEntryLast;
      size_t cInstances;
   };

   struct AfterExaminationForPossibleSplitting {
      TreeNode<true> * pTreeNodeChildren;
      FractionalDataType splitGain; // put this at the top so that our priority queue can access it directly without adding anything to the pointer (this is slightly more efficient on intel systems at least)
      ActiveDataType divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting afterExaminationForPossibleSplitting;

      static_assert(std::is_pod<BeforeExaminationForPossibleSplitting>::value, "BeforeSplit must be POD (Plain Old Data) if we are going to use it in a union!");
      static_assert(std::is_pod<AfterExaminationForPossibleSplitting>::value, "AfterSplit must be POD (Plain Old Data) if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;
   HistogramBucketVectorEntry<true> aHistogramBucketVectorEntry[1];

   EBM_INLINE size_t GetInstances() const {
      return m_UNION.beforeExaminationForPossibleSplitting.cInstances;
   }
   EBM_INLINE void SetInstances(size_t cInstances) {
      m_UNION.beforeExaminationForPossibleSplitting.cInstances = cInstances;
   }
};

template<>
class TreeNodeData<false> {
   // regression version of the TreeNodeData
public:

   struct BeforeExaminationForPossibleSplitting {
      const HistogramBucket<false> * pHistogramBucketEntryFirst;
      const HistogramBucket<false> * pHistogramBucketEntryLast;
   };

   struct AfterExaminationForPossibleSplitting {
      TreeNode<false> * pTreeNodeChildren;
      FractionalDataType splitGain; // put this at the top so that our priority queue can access it directly without adding anything to the pointer (this is slightly more efficient on intel systems at least)
      ActiveDataType divisionValue;
   };

   union TreeNodeDataUnion {
      // we can save precious L1 cache space by keeping only what we need
      BeforeExaminationForPossibleSplitting beforeExaminationForPossibleSplitting;
      AfterExaminationForPossibleSplitting afterExaminationForPossibleSplitting;

      static_assert(std::is_pod<BeforeExaminationForPossibleSplitting>::value, "BeforeSplit must be POD (Plain Old Data) if we are going to use it in a union!");
      static_assert(std::is_pod<AfterExaminationForPossibleSplitting>::value, "AfterSplit must be POD (Plain Old Data) if we are going to use it in a union!");
   };

   TreeNodeDataUnion m_UNION;

   size_t m_cInstances;
   HistogramBucketVectorEntry<false> aHistogramBucketVectorEntry[1];

   EBM_INLINE size_t GetInstances() const {
      return m_cInstances;
   }
   EBM_INLINE void SetInstances(size_t cInstances) {
      m_cInstances = cInstances;
   }
};

template<bool bClassification>
class TreeNode final : public TreeNodeData<bClassification> {
public:

   EBM_INLINE bool IsSplittable(size_t cInstancesRequiredForParentSplitMin) const {
      return this->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryLast != this->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryFirst && cInstancesRequiredForParentSplitMin <= this->GetInstances();
   }

   EBM_INLINE FractionalDataType EXTRACT_GAIN_BEFORE_SPLITTING() {
      EBM_ASSERT(this->m_UNION.afterExaminationForPossibleSplitting.splitGain <= 0);
      return this->m_UNION.afterExaminationForPossibleSplitting.splitGain;
   }

   EBM_INLINE void SPLIT_THIS_NODE() {
      this->m_UNION.afterExaminationForPossibleSplitting.splitGain = FractionalDataType { std::numeric_limits<FractionalDataType>::quiet_NaN() };
   }

   EBM_INLINE void INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED() {
      // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
      // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.afterExaminationForPossibleSplitting.splitGain and the m_UNION.beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
      this->m_UNION.afterExaminationForPossibleSplitting.splitGain = FractionalDataType { 0 };
   }

   EBM_INLINE bool WAS_THIS_NODE_SPLIT() const {
      return std::isnan(this->m_UNION.afterExaminationForPossibleSplitting.splitGain);
   }

   template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
   void ExamineNodeForPossibleSplittingAndDetermineBestSplitPoint(CachedTrainingThreadResources<bClassification> * const pCachedThreadResources, TreeNode<bClassification> * const pTreeNodeChildrenAvailableStorageSpaceCur, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
   ) {
      LOG(TraceLevelVerbose, "Entered SplitTreeNode: this=%p, pTreeNodeChildrenAvailableStorageSpaceCur=%p", static_cast<void *>(this), static_cast<void *>(pTreeNodeChildrenAvailableStorageSpaceCur));

      static_assert(IsClassification(compilerLearningTypeOrCountTargetClasses) == bClassification, "regression types must match");

      const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
      EBM_ASSERT(!GetTreeNodeSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerTreeNode = GetTreeNodeSize<bClassification>(cVectorLength);
      EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
      const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<bClassification>(cVectorLength);

      const HistogramBucket<bClassification> * pHistogramBucketEntryCur = this->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryFirst;
      const HistogramBucket<bClassification> * const pHistogramBucketEntryLast = this->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryLast;

      TreeNode<bClassification> * const pLeftChild1 = GetLeftTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
      pLeftChild1->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryFirst = pHistogramBucketEntryCur;
      TreeNode<bClassification> * const pRightChild1 = GetRightTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
      pRightChild1->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryLast = pHistogramBucketEntryLast;

      size_t cInstancesLeft = pHistogramBucketEntryCur->cInstancesInBucket;
      size_t cInstancesRight = this->GetInstances() - cInstancesLeft;

      HistogramBucketVectorEntry<bClassification> * const aSumHistogramBucketVectorEntryLeft = pCachedThreadResources->m_aSumHistogramBucketVectorEntry1;
      FractionalDataType * const aSumResidualErrorsRight = pCachedThreadResources->m_aSumResidualErrors2;
      HistogramBucketVectorEntry<bClassification> * const aSumHistogramBucketVectorEntryBest = pCachedThreadResources->m_aSumHistogramBucketVectorEntryBest;
      FractionalDataType BEST_nodeSplittingScore = 0;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         const FractionalDataType sumResidualErrorLeft = pHistogramBucketEntryCur->aHistogramBucketVectorEntry[iVector].sumResidualError;
         const FractionalDataType sumResidualErrorRight = this->aHistogramBucketVectorEntry[iVector].sumResidualError - sumResidualErrorLeft;

         BEST_nodeSplittingScore += EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorLeft, cInstancesLeft) + EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorRight, cInstancesRight);

         aSumHistogramBucketVectorEntryLeft[iVector].sumResidualError = sumResidualErrorLeft;
         aSumHistogramBucketVectorEntryBest[iVector].sumResidualError = sumResidualErrorLeft;
         aSumResidualErrorsRight[iVector] = sumResidualErrorRight;
         if(bClassification) {
            FractionalDataType sumDenominator1 = pHistogramBucketEntryCur->aHistogramBucketVectorEntry[iVector].GetSumDenominator();
            aSumHistogramBucketVectorEntryLeft[iVector].SetSumDenominator(sumDenominator1);
            aSumHistogramBucketVectorEntryBest[iVector].SetSumDenominator(sumDenominator1);
         }
      }

      EBM_ASSERT(0 <= BEST_nodeSplittingScore);
      const HistogramBucket<bClassification> * BEST_pHistogramBucketEntry = pHistogramBucketEntryCur;
      size_t BEST_cInstancesLeft = cInstancesLeft;
      for(pHistogramBucketEntryCur = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketEntryCur, 1); pHistogramBucketEntryLast != pHistogramBucketEntryCur; pHistogramBucketEntryCur = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, pHistogramBucketEntryCur, 1)) {
         ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pHistogramBucketEntryCur, aHistogramBucketsEndDebug);

         const size_t CHANGE_cInstances = pHistogramBucketEntryCur->cInstancesInBucket;
         cInstancesLeft += CHANGE_cInstances;
         cInstancesRight -= CHANGE_cInstances;

         FractionalDataType nodeSplittingScore = 0;
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            if(bClassification) {
               aSumHistogramBucketVectorEntryLeft[iVector].SetSumDenominator(aSumHistogramBucketVectorEntryLeft[iVector].GetSumDenominator() + pHistogramBucketEntryCur->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
            }

            const FractionalDataType CHANGE_sumResidualError = pHistogramBucketEntryCur->aHistogramBucketVectorEntry[iVector].sumResidualError;
            const FractionalDataType sumResidualErrorLeft = aSumHistogramBucketVectorEntryLeft[iVector].sumResidualError + CHANGE_sumResidualError;
            const FractionalDataType sumResidualErrorRight = aSumResidualErrorsRight[iVector] - CHANGE_sumResidualError;

            aSumHistogramBucketVectorEntryLeft[iVector].sumResidualError = sumResidualErrorLeft;
            aSumResidualErrorsRight[iVector] = sumResidualErrorRight;

            // TODO : we can make this faster by doing the division in ComputeNodeSplittingScore after we add all the numerators
            const FractionalDataType nodeSplittingScoreOneVector = EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorLeft, cInstancesLeft) + EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorRight, cInstancesRight);
            EBM_ASSERT(0 <= nodeSplittingScore);
            nodeSplittingScore += nodeSplittingScoreOneVector;
         }
         EBM_ASSERT(0 <= nodeSplittingScore);

         if(UNLIKELY(BEST_nodeSplittingScore < nodeSplittingScore)) {
            // TODO : randomly choose a node if BEST_entropyTotalChildren == entropyTotalChildren, but if there are 3 choice make sure that each has a 1/3 probability of being selected (same as interview question to select a random line from a file)
            BEST_nodeSplittingScore = nodeSplittingScore;
            BEST_pHistogramBucketEntry = pHistogramBucketEntryCur;
            BEST_cInstancesLeft = cInstancesLeft;
            memcpy(aSumHistogramBucketVectorEntryBest, aSumHistogramBucketVectorEntryLeft, sizeof(*aSumHistogramBucketVectorEntryBest) * cVectorLength);
         }
      }

      TreeNode<bClassification> * const pLeftChild = GetLeftTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);
      TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode);

      pLeftChild->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryLast = BEST_pHistogramBucketEntry;
      pLeftChild->SetInstances(BEST_cInstancesLeft);

      const HistogramBucket<bClassification> * const BEST_pHistogramBucketEntryNext = GetHistogramBucketByIndex<bClassification>(cBytesPerHistogramBucket, BEST_pHistogramBucketEntry, 1);
      ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, BEST_pHistogramBucketEntryNext, aHistogramBucketsEndDebug);

      pRightChild->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryFirst = BEST_pHistogramBucketEntryNext;
      size_t cInstancesParent = this->GetInstances();
      pRightChild->SetInstances(cInstancesParent - BEST_cInstancesLeft);

      FractionalDataType originalParentScore = 0;
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         pLeftChild->aHistogramBucketVectorEntry[iVector].sumResidualError = aSumHistogramBucketVectorEntryBest[iVector].sumResidualError;
         if(bClassification) {
            pLeftChild->aHistogramBucketVectorEntry[iVector].SetSumDenominator(aSumHistogramBucketVectorEntryBest[iVector].GetSumDenominator());
         }

         const FractionalDataType sumResidualErrorParent = this->aHistogramBucketVectorEntry[iVector].sumResidualError;
         originalParentScore += EbmStatistics::ComputeNodeSplittingScore(sumResidualErrorParent, cInstancesParent);

         pRightChild->aHistogramBucketVectorEntry[iVector].sumResidualError = sumResidualErrorParent - aSumHistogramBucketVectorEntryBest[iVector].sumResidualError;
         if(bClassification) {
            pRightChild->aHistogramBucketVectorEntry[iVector].SetSumDenominator(this->aHistogramBucketVectorEntry[iVector].GetSumDenominator() - aSumHistogramBucketVectorEntryBest[iVector].GetSumDenominator());
         }
      }



      // IMPORTANT!! : we need to finish all our calls that use this->m_UNION.beforeExaminationForPossibleSplitting BEFORE setting anything in m_UNION.afterExaminationForPossibleSplitting as we do below this comment!  The call above to this->GetInstances() needs to be done above these lines because it uses m_UNION.beforeExaminationForPossibleSplitting for classification!



      this->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren = pTreeNodeChildrenAvailableStorageSpaceCur;
      FractionalDataType splitGain = originalParentScore - BEST_nodeSplittingScore;
      if(UNLIKELY(std::isnan(splitGain))) {
         // it is possible that nodeSplittingScoreParent could reach infinity and BEST_nodeSplittingScore infinity, and the subtraction of those values leads to NaN
         // if gain became NaN via overlfow, that would be bad since we use NaN to indicate that a node has not been split
         splitGain = FractionalDataType { 0 };
      }
      this->m_UNION.afterExaminationForPossibleSplitting.splitGain = splitGain;
      this->m_UNION.afterExaminationForPossibleSplitting.divisionValue = (BEST_pHistogramBucketEntry->bucketValue + BEST_pHistogramBucketEntryNext->bucketValue) / 2;

      EBM_ASSERT(this->m_UNION.afterExaminationForPossibleSplitting.splitGain <= 0.0000000001); // within a set, no split should make our model worse.  It might in our validation set, but not within this set

      LOG(TraceLevelVerbose, "Exited SplitTreeNode: divisionValue=%zu, nodeSplittingScore=%" FractionalDataTypePrintf, static_cast<size_t>(this->m_UNION.afterExaminationForPossibleSplitting.divisionValue), this->m_UNION.afterExaminationForPossibleSplitting.splitGain);
   }

   // TODO: in theory, a malicious caller could overflow our stack if they pass us data that will grow a sufficiently deep tree.  Consider changing this recursive function to handle that
   // TODO: specialize this function for cases where we have hard coded vector lengths so that we don't have to pass in the cVectorLength parameter
   void Flatten(ActiveDataType ** const ppDivisions, FractionalDataType ** const ppValues, const size_t cVectorLength) const {
      // don't log this since we call it recursively.  Log where the root is called
      if(UNPREDICTABLE(WAS_THIS_NODE_SPLIT())) {
         EBM_ASSERT(!GetTreeNodeSizeOverflow<bClassification>(cVectorLength)); // we're accessing allocated memory
         const size_t cBytesPerTreeNode = GetTreeNodeSize<bClassification>(cVectorLength);
         const TreeNode<bClassification> * const pLeftChild = GetLeftTreeNodeChild<bClassification>(this->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode);
         pLeftChild->Flatten(ppDivisions, ppValues, cVectorLength);
         **ppDivisions = this->m_UNION.afterExaminationForPossibleSplitting.divisionValue;
         ++(*ppDivisions);
         const TreeNode<bClassification> * const pRightChild = GetRightTreeNodeChild<bClassification>(this->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode);
         pRightChild->Flatten(ppDivisions, ppValues, cVectorLength);
      } else {
         FractionalDataType * pValuesCur = *ppValues;
         FractionalDataType * const pValuesNext = pValuesCur + cVectorLength;
         *ppValues = pValuesNext;

         const HistogramBucketVectorEntry<bClassification> * pHistogramBucketVectorEntry = &this->aHistogramBucketVectorEntry[0];
         do {
            FractionalDataType smallChangeToModel;
            if(bClassification) {
               smallChangeToModel = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pHistogramBucketVectorEntry->sumResidualError, pHistogramBucketVectorEntry->GetSumDenominator());
            } else {
               smallChangeToModel = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pHistogramBucketVectorEntry->sumResidualError, this->GetInstances());
            }
            *pValuesCur = smallChangeToModel;

            ++pHistogramBucketVectorEntry;
            ++pValuesCur;
         } while(pValuesNext != pValuesCur);
      }
   }

   static_assert(std::is_pod<ActiveDataType>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");
};
static_assert(std::is_pod<TreeNode<false>>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");
static_assert(std::is_pod<TreeNode<true>>::value, "We want to keep our TreeNode compact and without a virtual pointer table for fitting in L1 cache as much as possible");

template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
bool GrowDecisionTree(CachedTrainingThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses, const size_t cHistogramBuckets, const HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBucket, const size_t cInstancesTotal, const HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntry, const size_t cTreeSplitsMax, const size_t cInstancesRequiredForParentSplitMin, SegmentedTensor<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet, FractionalDataType * const pTotalGain
#ifndef NDEBUG
   , const unsigned char * const aHistogramBucketsEndDebug
#endif // NDEBUG
) {
   LOG(TraceLevelVerbose, "Entered GrowDecisionTree");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);

   EBM_ASSERT(nullptr != pTotalGain);
   EBM_ASSERT(1 <= cInstancesTotal); // filter these out at the start where we can handle this case easily
   EBM_ASSERT(1 <= cHistogramBuckets); // cHistogramBuckets could only be zero if cInstancesTotal.  We should filter out that special case at our entry point though!!
   if(UNLIKELY(cInstancesTotal < cInstancesRequiredForParentSplitMin || 1 == cHistogramBuckets || 0 == cTreeSplitsMax)) {
      // there will be no splits at all

      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0))) {
         LOG(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 0)");
         return true;
      }

      // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength

      if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
         FractionalDataType smallChangeToModel = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(aSumHistogramBucketVectorEntry[0].sumResidualError, cInstancesTotal);
         FractionalDataType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
         pValues[0] = smallChangeToModel;
      } else {
         EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
         FractionalDataType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            FractionalDataType smallChangeToModel = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(aSumHistogramBucketVectorEntry[iVector].sumResidualError, aSumHistogramBucketVectorEntry[iVector].GetSumDenominator());
            aValues[iVector] = smallChangeToModel;
         }
      }

      LOG(TraceLevelVerbose, "Exited GrowDecisionTree via not enough data to split");
      *pTotalGain = 0;
      return false;
   }

   // there will be at least one split

   if(GetTreeNodeSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree GetTreeNodeSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)");
      return true; // we haven't accessed this TreeNode memory yet, so we don't know if it overflows yet
   }
   const size_t cBytesPerTreeNode = GetTreeNodeSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   EBM_ASSERT(!GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)); // we're accessing allocated memory
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);

retry_with_bigger_tree_node_children_array:
   size_t cBytesBuffer2 = pCachedThreadResources->GetThreadByteBuffer2Size();
   const size_t cBytesInitialNeededAllocation = 3 * cBytesPerTreeNode; // we need 1 TreeNode for the root, 1 for the left child of the root and 1 for the right child of the root
   if(cBytesBuffer2 < cBytesInitialNeededAllocation) {
      // TODO : we can eliminate this check as long as we ensure that the ThreadByteBuffer2 is always initialized to be equal to the size of three TreeNodes (left and right) == GET_SIZEOF_ONE_TREE_NODE_CHILDREN(cBytesPerTreeNode)
      if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesInitialNeededAllocation)) {
         LOG(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesInitialNeededAllocation)");
         return true;
      }
      cBytesBuffer2 = pCachedThreadResources->GetThreadByteBuffer2Size();
      EBM_ASSERT(cBytesInitialNeededAllocation <= cBytesBuffer2);
   }
   TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pRootTreeNode = static_cast<TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(pCachedThreadResources->GetThreadByteBuffer2());

   pRootTreeNode->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryFirst = aHistogramBucket;
   pRootTreeNode->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryLast = GetHistogramBucketByIndex<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cBytesPerHistogramBucket, aHistogramBucket, cHistogramBuckets - 1);
   ASSERT_BINNED_BUCKET_OK(cBytesPerHistogramBucket, pRootTreeNode->m_UNION.beforeExaminationForPossibleSplitting.pHistogramBucketEntryLast, aHistogramBucketsEndDebug);
   pRootTreeNode->SetInstances(cInstancesTotal);

   memcpy(&pRootTreeNode->aHistogramBucketVectorEntry[0], aSumHistogramBucketVectorEntry, cVectorLength * sizeof(*aSumHistogramBucketVectorEntry)); // copying existing mem

   pRootTreeNode->template ExamineNodeForPossibleSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, AddBytesTreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode, cBytesPerTreeNode), runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   if(UNPREDICTABLE(PREDICTABLE(1 == cTreeSplitsMax) || UNPREDICTABLE(2 == cHistogramBuckets))) {
      // there will be exactly 1 split, which is a special case that we can return faster without as much overhead as the multiple split case

      EBM_ASSERT(2 != cHistogramBuckets || !GetLeftTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode)->IsSplittable(cInstancesRequiredForParentSplitMin) && !GetRightTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode)->IsSplittable(cInstancesRequiredForParentSplitMin));

      if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1))) {
         LOG(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, 1)");
         return true;
      }

      ActiveDataType * pDivisions = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
      pDivisions[0] = pRootTreeNode->m_UNION.afterExaminationForPossibleSplitting.divisionValue;

      // we don't need to call EnsureValueCapacity because by default we start with a value capacity of 2 * cVectorLength

      // TODO : we don't need to get the right and left pointer from the root.. we know where they will be
      const TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pLeftChild = GetLeftTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode);
      const TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pRightChild = GetRightTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode);

      FractionalDataType * const aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
         aValues[0] = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pLeftChild->aHistogramBucketVectorEntry[0].sumResidualError, pLeftChild->GetInstances());
         aValues[1] = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(pRightChild->aHistogramBucketVectorEntry[0].sumResidualError, pRightChild->GetInstances());
      } else {
         EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
         for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
            aValues[iVector] = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pLeftChild->aHistogramBucketVectorEntry[iVector].sumResidualError, pLeftChild->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
            aValues[cVectorLength + iVector] = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(pRightChild->aHistogramBucketVectorEntry[iVector].sumResidualError, pRightChild->aHistogramBucketVectorEntry[iVector].GetSumDenominator());
         }
      }

      LOG(TraceLevelVerbose, "Exited GrowDecisionTree via one tree split");
      *pTotalGain = pRootTreeNode->EXTRACT_GAIN_BEFORE_SPLITTING();
      return false;
   }

   // it's very very likely that there will be more than 1 split below this point.  The only case where we wouldn't split below is if both our children nodes dont't have enough cases
   // to split, but that should be very rare

   // TODO: there are three types of queues that we should try out -> dyamically picking a stragety is a single predictable if statement, so shouldn't cause a lot of overhead
   //       1) When the data is the smallest(1-5 items), just iterate over all items in our TreeNode buffer looking for the best Node.  Zero the value on any nodes that have been removed from the queue.  For 1 or 2 instructions in the loop WITHOUT a branch we can probably save the pointer to the first TreeNode with data so that we can start from there next time we loop
   //       2) When the data is a tiny bit bigger and there are holes in our array of TreeNodes, we can maintain a pointer and value in a separate list and zip through the values and then go to the pointer to the best node.  Since the list is unordered, when we find a TreeNode to remove, we just move the last one into the hole
   //       3) The full fleged priority queue below
   size_t cSplits;
   try {
      std::priority_queue<TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> *, std::vector<TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>, CompareTreeNodeSplittingGain<IsClassification(compilerLearningTypeOrCountTargetClasses)>> * pBestTreeNodeToSplit = &pCachedThreadResources->m_bestTreeNodeToSplit;

      // it is ridiculous that we need to do this in order to clear the tree (there is no "clear" function), but inside this queue is a chunk of memory, and we want to ensure that the chunk of memory stays in L1 cache, so we pop all the previous garbage off instead of allocating a new one!
      while(!pBestTreeNodeToSplit->empty()) {
         pBestTreeNodeToSplit->pop();
      }

      cSplits = 0;
      TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pParentTreeNode = pRootTreeNode;

      // we skip 3 tree nodes.  The root, the left child of the root, and the right child of the root
      TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNodeChildrenAvailableStorageSpaceCur = AddBytesTreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pRootTreeNode, cBytesInitialNeededAllocation);

      FractionalDataType totalGain = 0;

      goto skip_first_push_pop;

      do {
         // there is no way to get the top and pop at the same time.. would be good to get a better queue, but our code isn't bottlenecked by it
         pParentTreeNode = pBestTreeNodeToSplit->top();
         pBestTreeNodeToSplit->pop();

      skip_first_push_pop:

         // ONLY AFTER WE'VE POPPED pParentTreeNode OFF the priority queue is it considered to have been split.  Calling SPLIT_THIS_NODE makes it formal
         totalGain += pParentTreeNode->EXTRACT_GAIN_BEFORE_SPLITTING();
         pParentTreeNode->SPLIT_THIS_NODE();

         TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pLeftChild = GetLeftTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pParentTreeNode->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode);
         if(pLeftChild->IsSplittable(cInstancesRequiredForParentSplitMin)) {
            TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNodeChildrenAvailableStorageSpaceNext = AddBytesTreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode << 1);
            if(cBytesBuffer2 < static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceNext) - reinterpret_cast<char *>(pRootTreeNode))) {
               if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)) {
                  LOG(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)");
                  return true;
               }
               goto retry_with_bigger_tree_node_children_array;
            }
            // the act of splitting it implicitly sets INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED because splitting sets splitGain to a non-NaN value
            pLeftChild->template ExamineNodeForPossibleSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, pTreeNodeChildrenAvailableStorageSpaceCur, runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
               , aHistogramBucketsEndDebug
#endif // NDEBUG
            );
            pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
            pBestTreeNodeToSplit->push(pLeftChild);
         } else {
            // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
            // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.afterExaminationForPossibleSplitting.splitGain and the m_UNION.beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
            pLeftChild->INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED();
         }

         TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pRightChild = GetRightTreeNodeChild<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pParentTreeNode->m_UNION.afterExaminationForPossibleSplitting.pTreeNodeChildren, cBytesPerTreeNode);
         if(pRightChild->IsSplittable(cInstancesRequiredForParentSplitMin)) {
            TreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)> * pTreeNodeChildrenAvailableStorageSpaceNext = AddBytesTreeNode<IsClassification(compilerLearningTypeOrCountTargetClasses)>(pTreeNodeChildrenAvailableStorageSpaceCur, cBytesPerTreeNode << 1);
            if(cBytesBuffer2 < static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceNext) - reinterpret_cast<char *>(pRootTreeNode))) {
               if(pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)) {
                  LOG(TraceLevelWarning, "WARNING GrowDecisionTree pCachedThreadResources->GrowThreadByteBuffer2(cBytesPerTreeNode)");
                  return true;
               }
               goto retry_with_bigger_tree_node_children_array;
            }
            // the act of splitting it implicitly sets INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED because splitting sets splitGain to a non-NaN value
            pRightChild->template ExamineNodeForPossibleSplittingAndDetermineBestSplitPoint<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, pTreeNodeChildrenAvailableStorageSpaceCur, runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
               , aHistogramBucketsEndDebug
#endif // NDEBUG
            );
            pTreeNodeChildrenAvailableStorageSpaceCur = pTreeNodeChildrenAvailableStorageSpaceNext;
            pBestTreeNodeToSplit->push(pRightChild);
         } else {
            // we aren't going to split this TreeNode because we can't.  We need to set the splitGain value here because otherwise it is filled with garbage that could be NaN (meaning the node was a branch)
            // we can't call INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED before calling SplitTreeNode because INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED sets m_UNION.afterExaminationForPossibleSplitting.splitGain and the m_UNION.beforeExaminationForPossibleSplitting values are needed if we had decided to call ExamineNodeForSplittingAndDetermineBestPossibleSplit
            pRightChild->INDICATE_THIS_NODE_EXAMINED_FOR_SPLIT_AND_REJECTED();
         }
         ++cSplits;
      } while(cSplits < cTreeSplitsMax && UNLIKELY(!pBestTreeNodeToSplit->empty()));
      // we DON'T need to call SetLeafAfterDone() on any items that remain in the pBestTreeNodeToSplit queue because everything in that queue has set a non-NaN nodeSplittingScore value

      // we might as well dump this value out to our pointer, even if later fail the function below.  If the function is failed, we make no guarantees about what we did with the value pointed to at *pTotalGain
      *pTotalGain = totalGain;
      EBM_ASSERT(static_cast<size_t>(reinterpret_cast<char *>(pTreeNodeChildrenAvailableStorageSpaceCur) - reinterpret_cast<char *>(pRootTreeNode)) <= cBytesBuffer2);
   } catch(...) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree exception");
      return true;
   }

   if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, cSplits))) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->SetCountDivisions(0, cSplits)");
      return true;
   }
   if(IsMultiplyError(cVectorLength, cSplits + 1)) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree IsMultiplyError(cVectorLength, cSplits + 1)");
      return true;
   }
   if(UNLIKELY(pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * (cSplits + 1)))) {
      LOG(TraceLevelWarning, "WARNING GrowDecisionTree pSmallChangeToModelOverwriteSingleSamplingSet->EnsureValueCapacity(cVectorLength * (cSplits + 1)");
      return true;
   }
   ActiveDataType * pDivisions = pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0);
   FractionalDataType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();

   LOG(TraceLevelVerbose, "Entered Flatten");
   pRootTreeNode->Flatten(&pDivisions, &pValues, cVectorLength);
   LOG(TraceLevelVerbose, "Exited Flatten");

   EBM_ASSERT(pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0) <= pDivisions);
   EBM_ASSERT(static_cast<size_t>(pDivisions - pSmallChangeToModelOverwriteSingleSamplingSet->GetDivisionPointer(0)) == cSplits);
   EBM_ASSERT(pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer() < pValues);
   EBM_ASSERT(static_cast<size_t>(pValues - pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer()) == cVectorLength * (cSplits + 1));

   LOG(TraceLevelVerbose, "Exited GrowDecisionTree via normal exit");
   return false;
}

// TODO : make variable ordering consistent with BinDataSet call below (put the feature first since that's a definition that happens before the training data set)
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
bool TrainZeroDimensional(CachedTrainingThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const SamplingMethod * const pTrainingSet, SegmentedTensor<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG(TraceLevelVerbose, "Entered TrainZeroDimensional");

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)) {
      // TODO : move this to initialization where we execute it only once (it needs to be in the feature combination loop though)
      LOG(TraceLevelWarning, "WARNING TODO fill this in");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pHistogramBucket = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesPerHistogramBucket));
   if(UNLIKELY(nullptr == pHistogramBucket)) {
      LOG(TraceLevelWarning, "WARNING TrainZeroDimensional nullptr == pHistogramBucket");
      return true;
   }
   memset(pHistogramBucket, 0, cBytesPerHistogramBucket);

   BinDataSetTrainingZeroDimensions<compilerLearningTypeOrCountTargetClasses>(pHistogramBucket, pTrainingSet, runtimeLearningTypeOrCountTargetClasses);

   const HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntry = &pHistogramBucket->aHistogramBucketVectorEntry[0];
   if(IsRegression(compilerLearningTypeOrCountTargetClasses)) {
      FractionalDataType smallChangeToModel = EbmStatistics::ComputeSmallChangeInRegressionPredictionForOneSegment(aSumHistogramBucketVectorEntry[0].sumResidualError, pHistogramBucket->cInstancesInBucket);
      FractionalDataType * pValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      pValues[0] = smallChangeToModel;
   } else {
      EBM_ASSERT(IsClassification(compilerLearningTypeOrCountTargetClasses));
      FractionalDataType * aValues = pSmallChangeToModelOverwriteSingleSamplingSet->GetValuePointer();
      for(size_t iVector = 0; iVector < cVectorLength; ++iVector) {
         FractionalDataType smallChangeToModel = EbmStatistics::ComputeSmallChangeInClassificationLogOddPredictionForOneSegment(aSumHistogramBucketVectorEntry[iVector].sumResidualError, aSumHistogramBucketVectorEntry[iVector].GetSumDenominator());
         aValues[iVector] = smallChangeToModel;
      }
   }

   LOG(TraceLevelVerbose, "Exited TrainZeroDimensional");
   return false;
}

// TODO : make variable ordering consistent with BinDataSet call below (put the feature first since that's a definition that happens before the training data set)
template<ptrdiff_t compilerLearningTypeOrCountTargetClasses>
bool TrainSingleDimensional(CachedTrainingThreadResources<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const pCachedThreadResources, const SamplingMethod * const pTrainingSet, const FeatureCombinationCore * const pFeatureCombination, const size_t cTreeSplitsMax, const size_t cInstancesRequiredForParentSplitMin, SegmentedTensor<ActiveDataType, FractionalDataType> * const pSmallChangeToModelOverwriteSingleSamplingSet, FractionalDataType * const pTotalGain, const ptrdiff_t runtimeLearningTypeOrCountTargetClasses) {
   LOG(TraceLevelVerbose, "Entered TrainSingleDimensional");

   EBM_ASSERT(1 == pFeatureCombination->m_cFeatures);
   size_t cTotalBuckets = pFeatureCombination->m_FeatureCombinationEntry[0].m_pFeature->m_cBins;

   const size_t cVectorLength = GET_VECTOR_LENGTH(compilerLearningTypeOrCountTargetClasses, runtimeLearningTypeOrCountTargetClasses);
   if(GetHistogramBucketSizeOverflow<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength)) {
      // TODO : move this to initialization where we execute it only once (it needs to be in the feature combination loop though)
      LOG(TraceLevelWarning, "WARNING TODO fill this in");
      return true;
   }
   const size_t cBytesPerHistogramBucket = GetHistogramBucketSize<IsClassification(compilerLearningTypeOrCountTargetClasses)>(cVectorLength);
   if(IsMultiplyError(cTotalBuckets, cBytesPerHistogramBucket)) {
      // TODO : move this to initialization where we execute it only once (it needs to be in the feature combination loop though)
      LOG(TraceLevelWarning, "WARNING TODO fill this in");
      return true;
   }
   const size_t cBytesBuffer = cTotalBuckets * cBytesPerHistogramBucket;
   HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aHistogramBuckets = static_cast<HistogramBucket<IsClassification(compilerLearningTypeOrCountTargetClasses)> *>(pCachedThreadResources->GetThreadByteBuffer1(cBytesBuffer));
   if(UNLIKELY(nullptr == aHistogramBuckets)) {
      LOG(TraceLevelWarning, "WARNING TrainSingleDimensional nullptr == aHistogramBuckets");
      return true;
   }
   // !!! VERY IMPORTANT: zero our one extra bucket for BuildFastTotals to use for multi-dimensional !!!!
   memset(aHistogramBuckets, 0, cBytesBuffer);

#ifndef NDEBUG
   const unsigned char * const aHistogramBucketsEndDebug = reinterpret_cast<unsigned char *>(aHistogramBuckets) + cBytesBuffer;
#endif // NDEBUG

   BinDataSetTraining<compilerLearningTypeOrCountTargetClasses, 1>(aHistogramBuckets, pFeatureCombination, pTrainingSet, runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   HistogramBucketVectorEntry<IsClassification(compilerLearningTypeOrCountTargetClasses)> * const aSumHistogramBucketVectorEntry = pCachedThreadResources->m_aSumHistogramBucketVectorEntry;
   memset(aSumHistogramBucketVectorEntry, 0, sizeof(*aSumHistogramBucketVectorEntry) * cVectorLength); // can't overflow, accessing existing memory

   size_t cHistogramBuckets = pFeatureCombination->m_FeatureCombinationEntry[0].m_pFeature->m_cBins;
   EBM_ASSERT(1 <= cHistogramBuckets); // this function can handle 1 == cBins even though that's a degenerate case that shouldn't be trained on (dimensions with 1 bin don't contribute anything since they always have the same value)
   size_t cInstancesTotal;
   cHistogramBuckets = CompressHistogramBuckets<compilerLearningTypeOrCountTargetClasses>(pTrainingSet, cHistogramBuckets, aHistogramBuckets, &cInstancesTotal, aSumHistogramBucketVectorEntry, runtimeLearningTypeOrCountTargetClasses
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   EBM_ASSERT(1 <= cInstancesTotal);
   EBM_ASSERT(1 <= cHistogramBuckets);

   bool bRet = GrowDecisionTree<compilerLearningTypeOrCountTargetClasses>(pCachedThreadResources, runtimeLearningTypeOrCountTargetClasses, cHistogramBuckets, aHistogramBuckets, cInstancesTotal, aSumHistogramBucketVectorEntry, cTreeSplitsMax, cInstancesRequiredForParentSplitMin, pSmallChangeToModelOverwriteSingleSamplingSet, pTotalGain
#ifndef NDEBUG
      , aHistogramBucketsEndDebug
#endif // NDEBUG
   );

   LOG(TraceLevelVerbose, "Exited TrainSingleDimensional");
   return bRet;
}

#endif // TREE_NODE_H
