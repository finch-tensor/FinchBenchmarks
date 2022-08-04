#include <taco/lower/mode_format_compressed.h>
#include "taco/lower/lowerer_impl.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/provenance_graph.h"
#include "taco/ir/ir.h"
#include "ir/ir_generators.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/simplify.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"
#include "mode_access.h"
#include "taco/util/collections.h"
#include "taco/util/env.h"

#include <tuple>

using namespace std;
using namespace taco::ir;
using taco::util::combine;

namespace taco {

class LowererImpl::Visitor : public IndexNotationVisitorStrict {
public:
  Visitor(LowererImpl* impl) : impl(impl) {}
  Stmt lower(IndexStmt stmt) {
    this->stmt = Stmt();
    impl->accessibleIterators.scope();
    IndexStmtVisitorStrict::visit(stmt);
    impl->accessibleIterators.unscope();
//    std::cout << "[lower] IndexStmt stmt: " << stmt << std::endl;
//    std::cout << "[lower] Lowered stmt: " << std::endl << this->stmt << std::endl;
    return this->stmt;
  }
  Expr lower(IndexExpr expr) {
    this->expr = Expr();
    IndexExprVisitorStrict::visit(expr);
    return this->expr;
  }
private:
  LowererImpl* impl;
  Expr expr;
  Stmt stmt;
  using IndexNotationVisitorStrict::visit;
  void visit(const AssignmentNode* node)    { stmt = impl->lowerAssignment(node); }
  void visit(const YieldNode* node)         { stmt = impl->lowerYield(node); }
  void visit(const ForallNode* node)        { stmt = impl->lowerForall(node); }
  void visit(const WhereNode* node)         { stmt = impl->lowerWhere(node); }
  void visit(const MultiNode* node)         { stmt = impl->lowerMulti(node); }
  void visit(const SuchThatNode* node)      { stmt = impl->lowerSuchThat(node); }
  void visit(const SequenceNode* node)      { stmt = impl->lowerSequence(node); }
  void visit(const AssembleNode* node)      { stmt = impl->lowerAssemble(node); }
  void visit(const AccessNode* node)        { expr = impl->lowerAccess(node); }
  void visit(const LiteralNode* node)       { expr = impl->lowerLiteral(node); }
  void visit(const NegNode* node)           { expr = impl->lowerNeg(node); }
  void visit(const AddNode* node)           { expr = impl->lowerAdd(node); }
  void visit(const SubNode* node)           { expr = impl->lowerSub(node); }
  void visit(const MulNode* node)           { expr = impl->lowerMul(node); }
  void visit(const DivNode* node)           { expr = impl->lowerDiv(node); }
  void visit(const SqrtNode* node)          { expr = impl->lowerSqrt(node); }
  void visit(const CastNode* node)          { expr = impl->lowerCast(node); }
  void visit(const CallIntrinsicNode* node) { expr = impl->lowerCallIntrinsic(node); }
  void visit(const CallNode* node)      { expr = impl->lowerTensorOp(node); }
  void visit(const ReductionNode* node)  {
    taco_ierror << "Reduction nodes not supported in concrete index notation";
  }
  void visit(const IndexVarNode* node)       { expr = impl->lowerIndexVar(node); }
};

LowererImpl::LowererImpl() : visitor(new Visitor(this)) {
}

static void createFillRegionVars(const map<TensorVar, Expr>& tensorVars,
                               map<Expr, vector<Expr>>* fillRegionVars) {
  for (auto& tensorVar : tensorVars) {
    Expr tensor = tensorVar.second;
    Expr fillRegionLength = Var::make(util::toString(tensor) + "_fill_len", Int());
    Expr fillRegionIndex = Var::make(util::toString(tensor) + "_fill_index", Int());
    fillRegionVars->insert({tensor, {fillRegionLength, fillRegionIndex}});
  }
}

static void createTensorTypeMap(const map<TensorVar, Expr>& tensorVars,
                                 map<Expr, Type>* tensorTypes) {
  for (auto& tensorVar : tensorVars) {
    Expr tensor = tensorVar.second;
    Type type = tensorVar.first.getType();
    tensorTypes->insert({tensor, type});
  }
}

static void createReverseTensorMap(const map<TensorVar, Expr>& tensorVars,
                                map<Expr, TensorVar>* tensorMap) {
  for (auto& tensorVar : tensorVars) {
    Expr tensor = tensorVar.second;
    TensorVar t = tensorVar.first;
    tensorMap->insert({tensor, t});
  }
}


static void createCapacityVars(const map<TensorVar, Expr>& tensorVars,
                               map<Expr, Expr>* capacityVars) {
  for (auto& tensorVar : tensorVars) {
    Expr tensor = tensorVar.second;
    Expr capacityVar = Var::make(util::toString(tensor) + "_capacity", Int());
    capacityVars->insert({tensor, capacityVar});
  }
}

static void createReducedValueVars(const vector<Access>& inputAccesses,
                                   map<Access, Expr>* reducedValueVars) {
  for (const auto& access : inputAccesses) {
    const TensorVar inputTensor = access.getTensorVar();
    const std::string name = inputTensor.getName() + "_val";
    const Datatype type = inputTensor.getType().getDataType();
    reducedValueVars->insert({access, Var::make(name, type)});
  }
}

static void getDependentTensors(IndexStmt stmt, std::set<TensorVar>& tensors) {
  std::set<TensorVar> prev;
  do {
    prev = tensors;
    match(stmt,
      function<void(const AssignmentNode*, Matcher*)>([&](
          const AssignmentNode* n, Matcher* m) {
        if (util::contains(tensors, n->lhs.getTensorVar())) {
          const auto arguments = getArguments(Assignment(n));
          tensors.insert(arguments.begin(), arguments.end());
        }
      })
    );
  } while (prev != tensors);
}

static bool needComputeValues(IndexStmt stmt, TensorVar tensor) {
  if (tensor.getType().getDataType() != Bool) {
    return true;
  }

  struct ReturnsTrue : public IndexExprRewriterStrict {
    void visit(const AccessNode* op) {
      if (op->isAccessingStructure || (
          op->tensorVar.getFormat().getModeFormats().back().isZeroless() &&
          equals(op->tensorVar.getFill(), Literal(false)))) {
        expr = op;
      }
    }

    void visit(const LiteralNode* op) {
      if (op->getDataType() == Bool && op->getVal<bool>()) {
        expr = op;
      }
    }

    void visit(const NegNode* op) {
      expr = rewrite(op->a);
    }

    void visit(const AddNode* op) {
      if (rewrite(op->a).defined() || rewrite(op->b).defined()) {
        expr = op;
      }
    }

    void visit(const MulNode* op) {
      if (rewrite(op->a).defined() && rewrite(op->b).defined()) {
        expr = op;
      }
    }

    void visit(const CastNode* op) {
      expr = rewrite(op->a);
    }

    void visit(const CallNode* op) {
      const auto annihilator = findProperty<Annihilator>(op->properties);

      if (!annihilator.defined() || !annihilator.positions().empty()) {
        return;
      }

      if (equals(annihilator.annihilator(), Literal(false))) {
        for (const auto& arg : op->args) {
          if (!rewrite(arg).defined()) {
            return;
          }
        }
        expr = op;
      } else {
        for (const auto& arg : op->args) {
          if (rewrite(arg).defined()) {
            expr = op;
            return;
          }
        }
      }
    }

    void visit(const SqrtNode* op) {}
    void visit(const SubNode* op) {}
    void visit(const DivNode* op) {}
    void visit(const CallIntrinsicNode* op) {}
    void visit(const ReductionNode* op) {}
    void visit(const IndexVarNode* op) {}
  };

  bool needComputeValue = false;
  match(stmt,
    function<void(const AssignmentNode*, Matcher*)>([&](
        const AssignmentNode* n, Matcher* m) {
      if (n->lhs.getTensorVar() == tensor &&
          !ReturnsTrue().rewrite(n->rhs).defined()) {
        needComputeValue = true;
      }
    })
  );

  return needComputeValue;
}

/// Returns true iff a result mode is assembled by inserting a sparse set of
/// result coordinates (e.g., compressed to dense).
static
bool hasSparseInserts(const std::vector<Iterator>& resultIterators,
                      const std::multimap<IndexVar, Iterator>& inputIterators) {
  for (const auto& resultIterator : resultIterators) {
    if (resultIterator.hasInsert()) {
      const auto indexVar = resultIterator.getIndexVar();
      const auto accessedInputs = inputIterators.equal_range(indexVar);
      for (auto inputIterator = accessedInputs.first;
           inputIterator != accessedInputs.second; ++inputIterator) {
        if (!inputIterator->second.isFull()) {
          return true;
        }
      }
    }
  }
  return false;
}

Stmt
LowererImpl::lower(IndexStmt stmt, string name,
                   bool assemble, bool compute, bool pack, bool unpack)
{
  this->assemble = assemble;
  this->compute = compute;
  definedIndexVarsOrdered = {};
  definedIndexVars = {};
  loopOrderAllowsShortCircuit = allForFreeLoopsBeforeAllReductionLoops(stmt);

  // Create result and parameter variables
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temporaries = getTemporaries(stmt);

  needCompute = {};
  if (generateAssembleCode()) {
    const auto attrQueryResults = getAttrQueryResults(stmt);
    needCompute.insert(attrQueryResults.begin(), attrQueryResults.end());
  }
  if (generateComputeCode()) {
    needCompute.insert(results.begin(), results.end());
  }
  getDependentTensors(stmt, needCompute);

  assembledByUngroupedInsert = util::toSet(
      getAssembledByUngroupedInsertion(stmt));

  // Create datastructure needed for temporary workspace hoisting/reuse
  temporaryInitialization = getTemporaryLocations(stmt);

  // Convert tensor results and arguments IR variables
  map<TensorVar, Expr> resultVars;
  vector<Expr> resultsIR = createVars(results, &resultVars, unpack);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createVars(arguments, &tensorVars, pack);

  // Create variables for index sets on result tensors.
  vector<Expr> indexSetArgs;
  for (auto& access : getResultAccesses(stmt).first) {
    // Any accesses that have index sets will be added.
    if (access.hasIndexSetModes()) {
      for (size_t i = 0; i < access.getIndexVars().size(); i++) {
        if (access.isModeIndexSet(i)) {
          auto t = access.getModeIndexSetTensor(i);
          if (tensorVars.count(t) == 0) {
            ir::Expr irVar = ir::Var::make(t.getName(), t.getType().getDataType(), true, true, pack);
            tensorVars.insert({t, irVar});
            indexSetArgs.push_back(irVar);
          }
        }
      }
    }
  }
  argumentsIR.insert(argumentsIR.begin(), indexSetArgs.begin(), indexSetArgs.end());

  // Create variables for temporaries
  // TODO Remove this
  for (auto& temp : temporaries) {
    ir::Expr irVar = ir::Var::make(temp.getName(), temp.getType().getDataType(),
                                   true, true);
    tensorVars.insert({temp, irVar});
  }

  // Create variables for keeping track of result values array capacity
  createCapacityVars(resultVars, &capacityVars);

  // Create variables used for iterating and keeping track of fill regions
  createFillRegionVars(tensorVars, &fillRegionVars);

  createTensorTypeMap(tensorVars, &tensorTypes);
  createReverseTensorMap(tensorVars, &tensorExprMap);

  // Create iterators
  iterators = Iterators(stmt, tensorVars);

  provGraph = ProvenanceGraph(stmt);

  for (const IndexVar& indexVar : provGraph.getAllIndexVars()) {
    if (iterators.modeIterators().count(indexVar)) {
      indexVarToExprMap.insert({indexVar, iterators.modeIterators()[indexVar].getIteratorVar()});
    }
    else {
      indexVarToExprMap.insert({indexVar, Var::make(indexVar.getName(), Int())});
    }
  }

  vector<Access> inputAccesses, resultAccesses;
  set<Access> reducedAccesses;
  inputAccesses = getArgumentAccesses(stmt);
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(stmt);

  // Create variables that represent the reduced values of duplicated tensor
  // components
  createReducedValueVars(inputAccesses, &reducedValueVars);

  map<TensorVar, Expr> scalars;

  // Define and initialize dimension variables
  set<TensorVar> temporariesSet(temporaries.begin(), temporaries.end());
  vector<IndexVar> indexVars = getIndexVars(stmt);
  for (auto& indexVar : indexVars) {
    Expr dimension;
    // getDimension extracts an Expr that holds the dimension
    // of a particular tensor mode. This Expr should be used as a loop bound
    // when iterating over the dimension of the target tensor.
    auto getDimension = [&](const TensorVar& tv, const Access& a, int mode) {
      // If the tensor mode is windowed, then the dimension for iteration is the bounds
      // of the window. Otherwise, it is the actual dimension of the mode.
      if (a.isModeWindowed(mode)) {
        // The mode value used to access .levelIterator is 1-indexed, while
        // the mode input to getDimension is 0-indexed. So, we shift it up by 1.
        auto iter = iterators.levelIterator(ModeAccess(a, mode+1));
        return ir::Div::make(ir::Sub::make(iter.getWindowUpperBound(), iter.getWindowLowerBound()), iter.getStride());
      } else if (a.isModeIndexSet(mode)) {
        // If the mode has an index set, then the dimension is the size of
        // the index set.
        return ir::Literal::make(a.getIndexSet(mode).size());
      } else {
        return GetProperty::make(tensorVars.at(tv), TensorProperty::Dimension, mode);
      }
    };
    match(stmt,
      function<void(const AssignmentNode*, Matcher*)>([&](
          const AssignmentNode* n, Matcher* m) {
        m->match(n->rhs);
        if (!dimension.defined()) {
          auto ivars = n->lhs.getIndexVars();
          auto tv = n->lhs.getTensorVar();
          int loc = (int)distance(ivars.begin(),
                                  find(ivars.begin(),ivars.end(), indexVar));
          if(!util::contains(temporariesSet, tv)) {
            dimension = getDimension(tv, n->lhs, loc);
          }
        }
      }),
      function<void(const AccessNode*)>([&](const AccessNode* n) {
        auto indexVars = n->indexVars;
        if (util::contains(indexVars, indexVar)) {
          int loc = (int)distance(indexVars.begin(),
                                  find(indexVars.begin(),indexVars.end(),
                                       indexVar));
          if(!util::contains(temporariesSet, n->tensorVar)) {
            dimension = getDimension(n->tensorVar, Access(n), loc);
          }
        }
      })
    );
    dimensions.insert({indexVar, dimension});
    underivedBounds.insert({indexVar, {ir::Literal::make(0), dimension}});
  }

  // Define and initialize scalar results and arguments
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(!util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        scalars.insert({result, tensorVars.at(result)});
        header.push_back(defineScalarVariable(result, true));
      }
    }
    for (auto& argument : arguments) {
      if (isScalar(argument.getType())) {
        taco_iassert(!util::contains(scalars, argument));
        taco_iassert(util::contains(tensorVars, argument));
        scalars.insert({argument, tensorVars.at(argument)});
        header.push_back(defineScalarVariable(argument, false));
      }
    }
  }

  // Allocate memory for scalar results
  if (generateAssembleCode()) {
    for (auto& result : results) {
      if (result.getOrder() == 0) {
        Expr resultIR = resultVars.at(result);
        Expr vals = GetProperty::make(resultIR, TensorProperty::Values);
        header.push_back(Allocate::make(vals, 1));
      }
    }
  }
  // Allocate and initialize append and insert mode indices
  Stmt initializeResults = initResultArrays(resultAccesses, inputAccesses,
                                            reducedAccesses);

  // Lower the index statement to compute and/or assemble
  Stmt body = lower(stmt);

  // Post-process result modes and allocate memory for values if necessary
  Stmt finalizeResults = finalizeResultArrays(resultAccesses);

  // Store scalar stack variables back to results
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        Expr resultIR = scalars.at(result);
        Expr varValueIR = tensorVars.at(result);
        Expr valuesArrIR = GetProperty::make(resultIR, TensorProperty::Values);
        footer.push_back(Store::make(valuesArrIR, 0, varValueIR, markAssignsAtomicDepth > 0, atomicParallelUnit));
      }
    }
  }

  // Create function
  return Function::make(name, resultsIR, argumentsIR,
                        Block::blanks(Block::make(header),
                                      initializeResults,
                                      body,
                                      finalizeResults,
                                      Block::make(footer)));
}


Stmt LowererImpl::lowerAssignment(Assignment assignment)
{
  taco_iassert(generateAssembleCode() || generateComputeCode());

  Stmt computeStmt;
  TensorVar result = assignment.getLhs().getTensorVar();
  Expr var = getTensorVar(result);

  const bool needComputeAssign = util::contains(needCompute, result);

  Expr rhs;
  if (needComputeAssign) {
    rhs = lower(assignment.getRhs());
  }

  // Assignment to scalar variables.
  if (isScalar(result.getType())) {
    if (needComputeAssign) {
      if (!assignment.getOperator().defined()) {
        computeStmt = Assign::make(var, rhs);
      }
      else {
        bool useAtomics = markAssignsAtomicDepth > 0 &&
                          !util::contains(whereTemps, result);
        if (isa<taco::Add>(assignment.getOperator())) {
          computeStmt = addAssign(var, rhs, useAtomics, atomicParallelUnit);
        } else {
          taco_iassert(isa<taco::Call>(assignment.getOperator()));

          Call op = to<Call>(assignment.getOperator());
          Expr assignOp = op.getFunc()({var, rhs});
          Stmt assign = Assign::make(var, assignOp, useAtomics,
                                     atomicParallelUnit);

          std::vector<Property> properties = op.getProperties();
          computeStmt = Block::make(assign, emitEarlyExit(var, properties));
        }
      }
    }
  }
  // Assignments to tensor variables (non-scalar).
  else {
    Expr values = getValuesArray(result);
    Expr loc = generateValueLocExpr(assignment.getLhs());

    std::vector<Stmt> accessStmts;

    if (isAssembledByUngroupedInsertion(result)) {
      std::vector<Expr> coords;
      Expr prevPos = 0;
      size_t i = 0;
      const auto resultIterators = getIterators(assignment.getLhs());
      for (const auto& it : resultIterators) {
        // TODO: Should only assemble levels that can be assembled together
        //if (it == this->nextTopResultIterator) {
        //  break;
        //}

        coords.push_back(getCoordinateVar(it));

        const auto yieldPos = it.getYieldPos(prevPos, coords);
        accessStmts.push_back(yieldPos.compute());
        Expr pos = it.getPosVar();
        accessStmts.push_back(VarDecl::make(pos, yieldPos[0]));

        if (generateAssembleCode()) {
          accessStmts.push_back(it.getInsertCoord(prevPos, pos, coords));
        }

        prevPos = pos;
        ++i;
      }
    }

    if (needComputeAssign && values.defined()) {
      if (!assignment.getOperator().defined()) {
        //TODO: DANIELBD
        if (getIterators(assignment.getLhs()).back().getPosIterKind() == taco_positer_kind::BYTE){
          auto charVals = ir::Cast::make(values, UInt8, true);
          auto addr = Load::make(charVals, loc, true);
          auto lhs = ir::Cast::make(addr, assignment.getLhs().getDataType(), true);

          computeStmt = Store::make(lhs, 0, rhs);
        } else {
          computeStmt = Store::make(values, loc, rhs);
        }
      }
      else {
        if (isa<taco::Add>(assignment.getOperator())) {
          computeStmt = compoundStore(values, loc, rhs, markAssignsAtomicDepth > 0, atomicParallelUnit);
        } else {

          taco_iassert(isa<taco::Call>(assignment.getOperator()));

          Call op = to<Call>(assignment.getOperator());
          Expr assignOp = op.getFunc()({Load::make(values, loc), rhs});
          computeStmt = Store::make(values, loc, assignOp,
                                    markAssignsAtomicDepth > 0 && !util::contains(whereTemps, result),
                                    atomicParallelUnit);

          std::vector<Property> properties = op.getProperties();
          computeStmt = Block::make(computeStmt, emitEarlyExit(Load::make(values, loc), properties));
        }
      }
      taco_iassert(computeStmt.defined());
    }

    if (!accessStmts.empty()) {
      accessStmts.push_back(computeStmt);
      computeStmt = Block::make(accessStmts);
    }
  }

  if (util::contains(guardedTemps, result) && result.getOrder() == 0) {
    Expr guard = tempToBitGuard[result];
    Stmt setGuard = Assign::make(guard, true, markAssignsAtomicDepth > 0,
                                 atomicParallelUnit);
    computeStmt = Block::make(computeStmt, setGuard);
  }

  Expr assembleGuard = generateAssembleGuard(assignment.getRhs());
  const bool assembleGuardTrivial = isa<ir::Literal>(assembleGuard);

  // TODO: If only assembling so defer allocating value memory to the end when
  //       we'll know exactly how much we need.
  bool temporaryWithSparseAcceleration = util::contains(tempToIndexList, result);
  if (generateComputeCode() && !temporaryWithSparseAcceleration) {
    taco_iassert(computeStmt.defined());
    return assembleGuardTrivial ? computeStmt : IfThenElse::make(assembleGuard,
                                                                 computeStmt);
  }

  if (temporaryWithSparseAcceleration) {
    taco_iassert(markAssignsAtomicDepth == 0)
      << "Parallel assembly of sparse accelerator not supported";

    Expr values = getValuesArray(result);
    Expr loc = generateValueLocExpr(assignment.getLhs());

    Expr bitGuardArr = tempToBitGuard.at(result);
    Expr indexList = tempToIndexList.at(result);
    Expr indexListSize = tempToIndexListSize.at(result);

    Stmt markBitGuardAsTrue = Store::make(bitGuardArr, loc, true);
    Stmt trackIndex = Store::make(indexList, indexListSize, loc);
    Expr incrementSize = ir::Add::make(indexListSize, 1);
    Stmt incrementStmt = Assign::make(indexListSize, incrementSize);

    Stmt firstWriteAtIndex = Block::make(trackIndex, markBitGuardAsTrue, incrementStmt);
    if (needComputeAssign && values.defined()) {
      Stmt initialStorage = computeStmt;
      if (assignment.getOperator().defined()) {
        // computeStmt is a compund stmt so we need to emit an initial store
        // into the temporary
        initialStorage =  Store::make(values, loc, rhs);
      }
      firstWriteAtIndex = Block::make(initialStorage, firstWriteAtIndex);
    }

    Expr readBitGuard = Load::make(bitGuardArr, loc);
    computeStmt = IfThenElse::make(ir::Neg::make(readBitGuard),
                                   firstWriteAtIndex, computeStmt);
  }

  return assembleGuardTrivial ? computeStmt : IfThenElse::make(assembleGuard,
                                                               computeStmt);
}


  Stmt LowererImpl::lowerYield(Yield yield) {
  std::vector<Expr> coords;
  for (auto& indexVar : yield.getIndexVars()) {
    coords.push_back(getCoordinateVar(indexVar));
  }
  Expr val = lower(yield.getExpr());
  return ir::Yield::make(coords, val);
}


pair<vector<Iterator>, vector<Iterator>>
LowererImpl::splitAppenderAndInserters(const vector<Iterator>& results) {
  vector<Iterator> appenders;
  vector<Iterator> inserters;

  // TODO: Choose insert when the current forall is nested inside a reduction
  for (auto& result : results) {
    if (isAssembledByUngroupedInsertion(result.getTensor())) {
      continue;
    }

    taco_iassert(result.hasAppend() || result.hasInsert())
        << "Results must support append or insert";

    if (result.hasAppend()) {
      appenders.push_back(result);
    }
    else {
      taco_iassert(result.hasInsert());
      inserters.push_back(result);
    }
  }

  return {appenders, inserters};
}

class AllFillsVisitor : public IndexNotationVisitor {

public:
    AllFillsVisitor() {}

    std::tuple<bool, std::vector<TensorVar>, std::vector<Access>> check(const IndexStmt& expr) {
      expr.accept(this);
      return {allFills, tensorFills, accesses};
    }

private:
    bool allFills = true;
    std::vector<TensorVar> tensorFills;
    std::vector<Access> accesses;

    using IndexNotationVisitor::visit;
    void visit(const AccessNode* node) {
      allFills = false;
      IndexNotationVisitor::visit(node);
    }

    void visit(const CallIntrinsicNode* node){
      if(node->func->getName() == "FillVariable"){
        auto arg = node->args[0];
        while(isa<CallIntrinsic>(arg)){
          taco_iassert(to<CallIntrinsic>(arg).getFunc().getName() == "FillVariable");
          arg = to<CallIntrinsic>(arg).getArgs()[0];
        }
        if (!isa<Access>(arg)) {
          std::cout << arg << std::endl;
        }
        taco_iassert(isa<Access>(arg));
        auto tensorVar = to<Access>(arg).getTensorVar();
        tensorFills.push_back(tensorVar);
        accesses.push_back(to<Access>(arg));
        return;
      }
      IndexNotationVisitor::visit(node);
    }
};

bool iteratorParentUpdatesFill(Iterator it){
  while (!it.isRoot()){
    if (it.updatesFillRegion()){
      return true;
    }
    it = it.getParent();
  }
  return false;
}

Stmt LowererImpl::lowerForall(Forall forall)
{
  bool hasExactBound = provGraph.hasExactBound(forall.getIndexVar());
  bool forallNeedsUnderivedGuards = !hasExactBound && emitUnderivedGuards;
  if (!ignoreVectorize && forallNeedsUnderivedGuards &&
      (forall.getParallelUnit() == ParallelUnit::CPUVector ||
       forall.getUnrollFactor() > 0)) {
    return lowerForallCloned(forall);
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel) {
    inParallelLoopDepth++;
  }

  // Recover any available parents that were not recoverable previously
  vector<Stmt> recoverySteps;
  for (const IndexVar& varToRecover : provGraph.newlyRecoverableParents(forall.getIndexVar(), definedIndexVars)) {
    // place pos guard
    if (forallNeedsUnderivedGuards && provGraph.isCoordVariable(varToRecover) &&
        provGraph.getChildren(varToRecover).size() == 1 &&
        provGraph.isPosVariable(provGraph.getChildren(varToRecover)[0])) {
      IndexVar posVar = provGraph.getChildren(varToRecover)[0];
      std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(posVar, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

      Expr minGuard = Lt::make(indexVarToExprMap[posVar], iterBounds[0]);
      Expr maxGuard = Gte::make(indexVarToExprMap[posVar], iterBounds[1]);
      Expr guardCondition = Or::make(minGuard, maxGuard);
      if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
        guardCondition = maxGuard;
      }

      ir::Stmt guard = ir::IfThenElse::make(guardCondition, ir::Continue::make());
      recoverySteps.push_back(guard);
    }

    Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    taco_iassert(indexVarToExprMap.count(varToRecover));
    recoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));

    // After we've recovered this index variable, some iterators are now
    // accessible for use when declaring locator access variables. So, generate
    // the accessors for those locator variables as part of the recovery process.
    // This is necessary after a fuse transformation, for example: If we fuse
    // two index variables (i, j) into f, then after we've generated the loop for
    // f, all locate accessors for i and j are now available for use.
    std::vector<Iterator> itersForVar;
    for (auto& iters : iterators.levelIterators()) {
      // Collect all level iterators that have locate and iterate over
      // the recovered index variable.
      if (iters.second.getIndexVar() == varToRecover && iters.second.hasLocate()) {
        itersForVar.push_back(iters.second);
      }
    }
    // Finally, declare all of the collected iterators' position access variables.
    recoverySteps.push_back(this->declLocatePosVars(itersForVar));

    // place underived guard
    std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    if (forallNeedsUnderivedGuards && underivedBounds.count(varToRecover) &&
        !provGraph.hasPosDescendant(varToRecover)) {

      // FIXME: [Olivia] Check this with someone
      // Removed underived guard if indexVar is bounded is divisible by its split child indexVar
      vector<IndexVar> children = provGraph.getChildren(varToRecover);
      bool hasDirectDivBound = false;
      std::vector<ir::Expr> iterBoundsInner = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

        for (auto& c: children) {
          if (provGraph.hasExactBound(c) && provGraph.derivationPath(varToRecover, c).size() == 2) {
              std::vector<ir::Expr> iterBoundsUnderivedChild = provGraph.deriveIterBounds(c, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
              if (iterBoundsUnderivedChild[1].as<ir::Literal>()->getValue<int>() % iterBoundsInner[1].as<ir::Literal>()->getValue<int>() == 0)
              hasDirectDivBound = true;
              break;
          }
      }
      if (!hasDirectDivBound) {
          Stmt guard = IfThenElse::make(Gte::make(indexVarToExprMap[varToRecover], underivedBounds[varToRecover][1]),
                                        Continue::make());
          recoverySteps.push_back(guard);
      }
    }

    // If this index variable was divided into multiple equal chunks, then we
    // must add an extra guard to make sure that further scheduling operations
    // on descendent index variables exceed the bounds of each equal portion of
    // the loop. For a concrete example, consider a loop of size 10 that is divided
    // into two equal components -- 5 and 5. If the loop is then transformed
    // with .split(..., 3), each inner chunk of 5 will be split into chunks of
    // 3. Without an extra guard, the second chunk of 3 in the first group of 5
    // may attempt to perform an iteration for the second group of 5, which is
    // incorrect.
    if (this->provGraph.isDivided(varToRecover)) {
      // Collect the children iteration variables.
      auto children = this->provGraph.getChildren(varToRecover);
      auto outer = children[0];
      auto inner = children[1];
      // Find the iteration bounds of the inner variable -- that is the size
      // that the outer loop was broken into.
      auto bounds = this->provGraph.deriveIterBounds(inner, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
      // Use the difference between the bounds to find the size of the loop.
      auto dimLen = ir::Sub::make(bounds[1], bounds[0]);
      // For a variable f divided into into f1 and f2, the guard ensures that
      // for iteration f, f should be within f1 * dimLen and (f1 + 1) * dimLen.
      auto guard = ir::Gte::make(this->indexVarToExprMap[varToRecover], ir::Mul::make(ir::Add::make(this->indexVarToExprMap[outer], 1), dimLen));
      recoverySteps.push_back(IfThenElse::make(guard, ir::Continue::make()));
    }
  }
  Stmt recoveryStmt = Block::make(recoverySteps);

  taco_iassert(!definedIndexVars.count(forall.getIndexVar()));
  definedIndexVars.insert(forall.getIndexVar());
  definedIndexVarsOrdered.push_back(forall.getIndexVar());

  if (forall.getParallelUnit() != ParallelUnit::NotParallel) {
    taco_iassert(!parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(!parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars[forall.getParallelUnit()] = forall.getIndexVar();
    vector<Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    parallelUnitSizes[forall.getParallelUnit()] = ir::Sub::make(bounds[1], bounds[0]);
  }

  MergeLattice caseLattice = MergeLattice::make(forall, iterators, provGraph, definedIndexVars, whereTempsToResult);
  vector<Access> resultAccesses;
  set<Access> reducedAccesses;
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(forall);

  // Pre-allocate/initialize memory of value arrays that are full below this
  // loops index variable
  Stmt preInitValues = initResultArrays(forall.getIndexVar(), resultAccesses,
                                        getArgumentAccesses(forall),
                                        reducedAccesses);

  // Emit temporary initialization if forall is sequential or parallelized by cpu threads and leads to a where statement
  // This is for workspace hoisting by 1-level
  vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
  auto temp = temporaryInitialization.find(forall);
  if (temp != temporaryInitialization.end() && forall.getParallelUnit() == ParallelUnit::NotParallel && !isScalar(temp->second.getTemporary().getType()))
    temporaryValuesInitFree = codeToInitializeTemporary(temp->second);
  else if (temp != temporaryInitialization.end() && forall.getParallelUnit() == ParallelUnit::CPUThread && !isScalar(temp->second.getTemporary().getType())) {
    temporaryValuesInitFree = codeToInitializeTemporaryParallel(temp->second, forall.getParallelUnit());
  }

  auto fillsInfo = AllFillsVisitor().check(forall.getStmt());

  Stmt loops;
  // Emit a loop that iterates over over a single iterator (optimization)
  if (caseLattice.iterators().size() == 1 && caseLattice.iterators()[0].isUnique() &&
      ((!iteratorParentUpdatesFill(caseLattice.iterators()[0]) && std::get<1>(fillsInfo).empty()) || 
       caseLattice.iterators()[0].isDimensionIterator())) {
//    std::cout << "SINGLE ITERATOR " << caseLattice.iterators()[0] << endl;
    MergeLattice loopLattice = caseLattice.getLoopLattice();

    MergePoint point = loopLattice.points()[0];
    Iterator iterator = loopLattice.iterators()[0];

    vector<Iterator> locators = point.locators();
    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(point.results());

    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(iterator.getIndexVar());
    IndexVar posDescendant;
    bool hasPosDescendant = false;
    if (!underivedAncestors.empty()) {
      hasPosDescendant = provGraph.getPosIteratorFullyDerivedDescendant(underivedAncestors[0], &posDescendant);
    }

    bool isWhereProducer = false;
    vector<Iterator> results = point.results();
    for (Iterator result : results) {
      for (auto it = tensorVars.begin(); it != tensorVars.end(); it++) {
        if (it->second == result.getTensor()) {
          if (whereTempsToResult.count(it->first)) {
            isWhereProducer = true;
            break;
          }
        }
      }
    }

    // For now, this only works when consuming a single workspace.
    //bool canAccelWithSparseIteration = inParallelLoopDepth == 0 && provGraph.isFullyDerived(iterator.getIndexVar()) &&
    //                                   iterator.isDimensionIterator() && locators.size() == 1;
    bool canAccelWithSparseIteration =
        provGraph.isFullyDerived(iterator.getIndexVar()) &&
        iterator.isDimensionIterator() && locators.size() == 1;
    if (canAccelWithSparseIteration) {
      bool indexListsExist = false;
      // We are iterating over a dimension and locating into a temporary with a tracker to keep indices. Instead, we
      // can just iterate over the indices and locate into the dense workspace.
      for (auto it = tensorVars.begin(); it != tensorVars.end(); ++it) {
        if (it->second == locators[0].getTensor() && util::contains(tempToIndexList, it->first)) {
          indexListsExist = true;
          break;
        }
      }
      canAccelWithSparseIteration &= indexListsExist;
    }

    if (!isWhereProducer && hasPosDescendant && underivedAncestors.size() > 1 && provGraph.isPosVariable(iterator.getIndexVar()) && posDescendant == forall.getIndexVar()) {
      loops = lowerForallFusedPosition(forall, iterator, locators, inserters, appenders, caseLattice,
                                       reducedAccesses, recoveryStmt);
    }
    else if (canAccelWithSparseIteration) {
      loops = lowerForallDenseAcceleration(forall, locators, inserters, appenders, caseLattice, reducedAccesses, recoveryStmt);
    }
    // Emit dimension coordinate iteration loop
    else if (iterator.isDimensionIterator()) {
      loops = lowerForallDimension(forall, point.locators(), inserters, appenders, caseLattice,
                                   reducedAccesses, recoveryStmt);
    }
    // Emit position iteration loop
    else if (iterator.hasPosIter()) {
      loops = lowerForallPosition(forall, iterator, locators, inserters, appenders, caseLattice,
                                  reducedAccesses, recoveryStmt);
    }
    // Emit coordinate iteration loop
    else {
      taco_iassert(iterator.hasCoordIter());
      loops = lowerForallCoordinate(forall, iterator, locators, inserters, appenders, caseLattice,
                                    reducedAccesses, recoveryStmt);
    }
  }
  // Emit general loops to merge multiple iterators
  else {
    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
    taco_iassert(underivedAncestors.size() == 1); // TODO: add support for fused coordinate of pos loop
    loops = lowerMergeLattice(caseLattice, underivedAncestors[0],
                              forall.getStmt(), reducedAccesses);
  }
//  taco_iassert(loops.defined());

  if (!generateComputeCode() && !hasStores(loops)) {
    // If assembly loop does not modify output arrays, then it can be safely
    // omitted.
    loops = Stmt();
  }
  definedIndexVars.erase(forall.getIndexVar());
  definedIndexVarsOrdered.pop_back();
  if (forall.getParallelUnit() != ParallelUnit::NotParallel) {
    inParallelLoopDepth--;
    taco_iassert(parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars.erase(forall.getParallelUnit());
    parallelUnitSizes.erase(forall.getParallelUnit());
  }
  return Block::blanks(preInitValues,
                       temporaryValuesInitFree[0],
                       loops,
                       temporaryValuesInitFree[1]);
}

Stmt LowererImpl::lowerForallCloned(Forall forall) {
  // want to emit guards outside of loop to prevent unstructured loop exits

  // construct guard
  // underived or pos variables that have a descendant that has not been defined yet
  vector<IndexVar> varsWithGuard;
  for (auto var : provGraph.getAllIndexVars()) {
    if (provGraph.isRecoverable(var, definedIndexVars)) {
      continue; // already recovered
    }
    if (provGraph.isUnderived(var) && !provGraph.hasPosDescendant(var)) { // if there is pos descendant then will be guarded already
      varsWithGuard.push_back(var);
    }
    else if (provGraph.isPosVariable(var)) {
      // if parent is coord then this is variable that will be guarded when indexing into coord array
      if(provGraph.getParents(var).size() == 1 && provGraph.isCoordVariable(provGraph.getParents(var)[0])) {
        varsWithGuard.push_back(var);
      }
    }
  }

  // determine min and max values for vars given already defined variables.
  // we do a recovery where we fill in undefined variables with either 0's or the max of their iteration
  std::map<IndexVar, Expr> minVarValues;
  std::map<IndexVar, Expr> maxVarValues;
  set<IndexVar> definedForGuard = definedIndexVars;
  vector<Stmt> guardRecoverySteps;
  Expr maxOffset = 0;
  bool setMaxOffset = false;

  for (auto var : varsWithGuard) {
    std::vector<IndexVar> currentDefinedVarOrder = definedIndexVarsOrdered; // TODO: get defined vars at time of this recovery

    std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
    std::map<IndexVar, Expr> maxChildValues = indexVarToExprMap;

    for (auto child : provGraph.getFullyDerivedDescendants(var)) {
      if (!definedIndexVars.count(child)) {
        std::vector<ir::Expr> childBounds = provGraph.deriveIterBounds(child, currentDefinedVarOrder, underivedBounds, indexVarToExprMap, iterators);

        minChildValues[child] = childBounds[0];
        maxChildValues[child] = childBounds[1];

        // recover new parents
        for (const IndexVar& varToRecover : provGraph.newlyRecoverableParents(child, definedForGuard)) {
          Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                          minChildValues, iterators);
          Expr maxRecoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                             maxChildValues, iterators);
          if (!setMaxOffset) { // TODO: work on simplifying this
            maxOffset = ir::Add::make(maxOffset, ir::Sub::make(maxRecoveredValue, recoveredValue));
            setMaxOffset = true;
          }
          taco_iassert(indexVarToExprMap.count(varToRecover));

          guardRecoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
          definedForGuard.insert(varToRecover);
        }
        definedForGuard.insert(child);
      }
    }

    minVarValues[var] = provGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, minChildValues, iterators);
    maxVarValues[var] = provGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, maxChildValues, iterators);
  }

  // Build guards
  Expr guardCondition;
  for (auto var : varsWithGuard) {
    std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

    Expr minGuard = Lt::make(minVarValues[var], iterBounds[0]);
    Expr maxGuard = Gte::make(ir::Add::make(maxVarValues[var], ir::simplify(maxOffset)), iterBounds[1]);
    Expr guardConditionCurrent = Or::make(minGuard, maxGuard);

    if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
      guardConditionCurrent = maxGuard;
    }

    if (guardCondition.defined()) {
      guardCondition = Or::make(guardConditionCurrent, guardCondition);
    }
    else {
      guardCondition = guardConditionCurrent;
    }
  }

  Stmt unvectorizedLoop;
  // build loop with guards (not vectorized)
  if (!varsWithGuard.empty()) {
    ignoreVectorize = true;
    unvectorizedLoop = lowerForall(forall);
    ignoreVectorize = false;
  }

  // build loop without guards
  emitUnderivedGuards = false;
  Stmt vectorizedLoop = lowerForall(forall);
  emitUnderivedGuards = true;

  // return guarded loops
  return Block::make(Block::make(guardRecoverySteps), IfThenElse::make(guardCondition, unvectorizedLoop, vectorizedLoop));
}

Stmt LowererImpl::searchForFusedPositionStart(Forall forall, Iterator posIterator) {
  vector<Stmt> searchForUnderivedStart;
  vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
  ir::Expr last_block_start_temporary;
  for (int i = (int) underivedAncestors.size() - 2; i >= 0; i--) {
    Iterator posIteratorLevel = posIterator;
    for (int j = (int) underivedAncestors.size() - 2; j > i; j--) { // take parent of iterator enough times to get correct level
      posIteratorLevel = posIteratorLevel.getParent();
    }

    // want to get size of pos array not of crd_array
    ir::Expr parentSize = 1; // to find size of segment walk down sizes of iterator chain
    Iterator rootIterator = posIterator;
    while (!rootIterator.isRoot()) {
      rootIterator = rootIterator.getParent();
    }
    while (rootIterator.getChild() != posIteratorLevel) {
      rootIterator = rootIterator.getChild();
      if (rootIterator.hasAppend()) {
        parentSize = rootIterator.getSize(parentSize);
      } else if (rootIterator.hasInsert()) {
        parentSize = ir::Mul::make(parentSize, rootIterator.getWidth());
      }
    }

    // emit bounds search on cpu just bounds, on gpu search in blocks
    if (parallelUnitIndexVars.count(ParallelUnit::GPUBlock)) {
      Expr values_per_block;
      {
        // we do a recovery where we fill in undefined variables with 0's to get start target (just like for vector guards)
        std::map<IndexVar, Expr> zeroedChildValues = indexVarToExprMap;
        zeroedChildValues[parallelUnitIndexVars[ParallelUnit::GPUBlock]] = 1;
        set<IndexVar> zeroDefinedIndexVars = {parallelUnitIndexVars[ParallelUnit::GPUBlock]};
        for (IndexVar child : provGraph.getFullyDerivedDescendants(posIterator.getIndexVar())) {
          if (child != parallelUnitIndexVars[ParallelUnit::GPUBlock]) {
            zeroedChildValues[child] = 0;

            // recover new parents
            for (const IndexVar &varToRecover : provGraph.newlyRecoverableParents(child, zeroDefinedIndexVars)) {
              Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                              zeroedChildValues, iterators);
              taco_iassert(indexVarToExprMap.count(varToRecover));
              zeroedChildValues[varToRecover] = recoveredValue;
              zeroDefinedIndexVars.insert(varToRecover);
              if (varToRecover == posIterator.getIndexVar()) {
                break;
              }
            }
            zeroDefinedIndexVars.insert(child);
          }
        }
        values_per_block = zeroedChildValues[posIterator.getIndexVar()];
      }

      IndexVar underived = underivedAncestors[i];
      ir::Expr blockStarts_temporary = ir::Var::make(underived.getName() + "_blockStarts",
                                                     getCoordinateVar(underived).type(), true, false);
      header.push_back(ir::VarDecl::make(blockStarts_temporary, 0));
      header.push_back(
              Allocate::make(blockStarts_temporary, ir::Add::make(parallelUnitSizes[ParallelUnit::GPUBlock], 1)));
      footer.push_back(Free::make(blockStarts_temporary));


      Expr blockSize;
      if (parallelUnitSizes.count(ParallelUnit::GPUThread)) {
        blockSize = parallelUnitSizes[ParallelUnit::GPUThread];
        if (parallelUnitSizes.count(ParallelUnit::GPUWarp)) {
          blockSize = ir::Mul::make(blockSize, parallelUnitSizes[ParallelUnit::GPUWarp]);
        }
      } else {
        std::vector<IndexVar> definedIndexVarsMatched = definedIndexVarsOrdered;
        // find sub forall that tells us block size
        match(forall.getStmt(),
              function<void(const ForallNode *, Matcher *)>([&](
                      const ForallNode *n, Matcher *m) {
                if (n->parallel_unit == ParallelUnit::GPUThread) {
                  vector<Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsMatched,
                                                                   underivedBounds, indexVarToExprMap, iterators);
                  blockSize = ir::Sub::make(bounds[1], bounds[0]);
                }
                definedIndexVarsMatched.push_back(n->indexVar);
              })
        );
      }
      taco_iassert(blockSize.defined());

      if (i == (int) underivedAncestors.size() - 2) {
        std::vector<Expr> args = {
                posIteratorLevel.getMode().getModePack().getArray(0), // array
                blockStarts_temporary, // results
                ir::Literal::zero(posIteratorLevel.getBeginVar().type()), // arrayStart
                parentSize, // arrayEnd
                values_per_block, // values_per_block
                blockSize, // block_size
                parallelUnitSizes[ParallelUnit::GPUBlock] // num_blocks
        };
        header.push_back(ir::Assign::make(blockStarts_temporary,
                                          ir::Call::make("taco_binarySearchBeforeBlockLaunch", args,
                                                         getCoordinateVar(underived).type())));
      }
      else {
        std::vector<Expr> args = {
                posIteratorLevel.getMode().getModePack().getArray(0), // array
                blockStarts_temporary, // results
                ir::Literal::zero(posIteratorLevel.getBeginVar().type()), // arrayStart
                parentSize, // arrayEnd
                last_block_start_temporary, // targets
                blockSize, // block_size
                parallelUnitSizes[ParallelUnit::GPUBlock] // num_blocks
        };
        header.push_back(ir::Assign::make(blockStarts_temporary,
                                          ir::Call::make("taco_binarySearchIndirectBeforeBlockLaunch", args,
                                                         getCoordinateVar(underived).type())));
      }
      searchForUnderivedStart.push_back(VarDecl::make(posIteratorLevel.getBeginVar(),
                                                      ir::Load::make(blockStarts_temporary,
                                                                     indexVarToExprMap[parallelUnitIndexVars[ParallelUnit::GPUBlock]])));
      searchForUnderivedStart.push_back(VarDecl::make(posIteratorLevel.getEndVar(),
                                                      ir::Load::make(blockStarts_temporary, ir::Add::make(
                                                              indexVarToExprMap[parallelUnitIndexVars[ParallelUnit::GPUBlock]],
                                                              1))));
      last_block_start_temporary = blockStarts_temporary;
    } else {
      header.push_back(VarDecl::make(posIteratorLevel.getBeginVar(), ir::Literal::zero(posIteratorLevel.getBeginVar().type())));
      header.push_back(VarDecl::make(posIteratorLevel.getEndVar(), parentSize));
    }

    // we do a recovery where we fill in undefined variables with 0's to get start target (just like for vector guards)
    Expr underivedStartTarget;
    if (i == (int) underivedAncestors.size() - 2) {
      std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
      set<IndexVar> minDefinedIndexVars = definedIndexVars;
      minDefinedIndexVars.erase(forall.getIndexVar());

      for (IndexVar child : provGraph.getFullyDerivedDescendants(posIterator.getIndexVar())) {
        if (!minDefinedIndexVars.count(child)) {
          std::vector<ir::Expr> childBounds = provGraph.deriveIterBounds(child, definedIndexVarsOrdered,
                                                                         underivedBounds,
                                                                         indexVarToExprMap, iterators);
          minChildValues[child] = childBounds[0];

          // recover new parents
          for (const IndexVar &varToRecover : provGraph.newlyRecoverableParents(child, minDefinedIndexVars)) {
            Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                            minChildValues, iterators);
            taco_iassert(indexVarToExprMap.count(varToRecover));
            searchForUnderivedStart.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
            minDefinedIndexVars.insert(varToRecover);
            if (varToRecover == posIterator.getIndexVar()) {
              break;
            }
          }
          minDefinedIndexVars.insert(child);
        }
      }
      underivedStartTarget = indexVarToExprMap[posIterator.getIndexVar()];
    }
    else {
      underivedStartTarget = this->iterators.modeIterator(underivedAncestors[i+1]).getPosVar();
    }

    vector<Expr> binarySearchArgs = {
            posIteratorLevel.getMode().getModePack().getArray(0), // array
            posIteratorLevel.getBeginVar(), // arrayStart
            posIteratorLevel.getEndVar(), // arrayEnd
            underivedStartTarget // target
    };
    Expr posVarUnknown = this->iterators.modeIterator(underivedAncestors[i]).getPosVar();
    searchForUnderivedStart.push_back(ir::VarDecl::make(posVarUnknown,
                                                        ir::Call::make("taco_binarySearchBefore", binarySearchArgs,
                                                                       getCoordinateVar(underivedAncestors[i]).type())));
    Stmt locateCoordVar;
    if (posIteratorLevel.getParent().hasPosIter()) {
      locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], ir::Load::make(posIteratorLevel.getParent().getMode().getModePack().getArray(1), posVarUnknown));
    }
    else {
      locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], posVarUnknown);
    }
    searchForUnderivedStart.push_back(locateCoordVar);
  }
  return ir::Block::make(searchForUnderivedStart);
}

Stmt LowererImpl::lowerForallDimension(Forall forall,
                                       vector<Iterator> locators,
                                       vector<Iterator> inserters,
                                       vector<Iterator> appenders,
                                       MergeLattice caseLattice,
                                       set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
    atomicParallelUnit = forall.getParallelUnit();
  }



  Stmt body = lowerForallBody(coordinate, forall.getStmt(), locators, inserters,
                              appenders, caseLattice, reducedAccesses);

  std::vector<Stmt> fillUpdates;
  auto fillsResult = AllFillsVisitor().check(forall);
  bool canUseCoord = true;
  for (size_t i=0; i<std::get<1>(fillsResult).size(); i++){
      auto& tensor = std::get<1>(fillsResult)[i];
      auto& access = std::get<2>(fillsResult)[i];
      auto accessIters = getIterators(access);
      auto shape = tensor.getType().getShape();
      if (shape.getOrder() != (int)accessIters.size()){
        canUseCoord = false;
        break;
      }
      bool hasUpdateFill = false;
      for (size_t j=0; j< accessIters.size(); j++){
        auto& it = accessIters[j];
        if (it.updatesFillRegion()){
          hasUpdateFill = true;
          if (j+1 >= accessIters.size() || !shape.getDimension(j+1).isFixed()){
            // std::cout << "Dimension " << j+1 << " not fixed or out of bounds" << std::endl;
            canUseCoord = false;
            break;
          }
          Expr values = getValuesArrayFromIterator(it);
          ModeFunction posAccess = it.posAccess(positions(it),
                                                coordinates(it),
                                                values, tensorTypes[it.getTensor()].getDataType());
          ModeFunction fillUpdate = it.getFillRegion(it.getPosVar(),
                                                     coordinates(it),
                                                     values, tensorTypes[it.getTensor()].getDataType());
          if (posAccess.defined() &&
              posAccess.getResults().size() > 2 &&
              !(posAccess[3].as<ir::Literal>() && posAccess[3].as<ir::Literal>()->equalsScalar(shape.getDimension(j+1).getSize()))){
            // std::cout << "Length " << posAccess[3].as<ir::Literal>()->getIntValue() << " not equal to lower dim size: " << shape.getDimension(j+1).getSize() << std::endl;
            canUseCoord = false;
          }
          if (fillUpdate.defined() &&
              fillUpdate.getResults().size() == 3 &&
              !(fillUpdate[1].as<ir::Literal>() && fillUpdate[1].as<ir::Literal>()->equalsScalar(shape.getDimension(j+1).getSize()))){
            // std::cout << "Length " << fillUpdate[1].as<ir::Literal>()->getIntValue() << " not equal to lower dim size: " << shape.getDimension(j+1).getSize() << std::endl;
            canUseCoord = false;
          }

        }
      }
      if (!canUseCoord) break;

      if (hasUpdateFill){
        auto fillRegion = GetProperty::make(getTensorVar(tensor), TensorProperty::FillRegion);
        auto fillVariable = GetProperty::make(getTensorVar(tensor), TensorProperty::FillValue);
        fillUpdates.push_back(Assign::make(fillVariable, Load::make(fillRegion, coordinate)));
      }
  }
  if (canUseCoord){
    if (!fillUpdates.empty()){
      auto block = const_cast<ir::Block*>(body.as<ir::Block>());
      auto back_elem = block->contents.back();
      block->contents.pop_back();
      for (auto& update : fillUpdates){
        block->contents.push_back(update);
      }
      block->contents.push_back(back_elem);

      // std::cout << "BODY: " << body << std::endl;
      // body = Block::make(body, Block::make(fillUpdates));
    } else canUseCoord = false;
  } else {
    body = Block::make(body,codeToUpdateFills(forall.getStmt(), coordinate, forall.getIndexVar(), {}, {}));
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  body = Block::make({recoveryStmt, body});

  Stmt posAppend = generateAppendPositions(appenders);

  // Emit loop with preamble and postamble
  std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != ParallelUnit::NotParallel
            && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

  if (canUseCoord){
    // Specialize for bounds in this specific case
    auto shape = std::get<1>(fillsResult)[0].getType().getShape(); 
    return Block::blanks(For::make(coordinate, 0, (int)shape.getDimension(shape.getOrder()-1).getSize(), 1, body,
                              kind,
                              ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                    posAppend);

  } else {
    return Block::blanks(For::make(coordinate, bounds[0], bounds[1], 1, body,
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                       posAppend);
  }
}

  Stmt LowererImpl::lowerForallDenseAcceleration(Forall forall,
                                                 vector<Iterator> locators,
                                                 vector<Iterator> inserters,
                                                 vector<Iterator> appenders,
                                                 MergeLattice caseLattice,
                                                 set<Access> reducedAccesses,
                                                 ir::Stmt recoveryStmt)
  {
    taco_iassert(locators.size() == 1) << "Optimizing a dense workspace is only supported when the consumer is the only RHS tensor";
    taco_iassert(provGraph.isFullyDerived(forall.getIndexVar())) << "Sparsely accelerating a dense workspace only works with fully derived index vars";
    taco_iassert(forall.getParallelUnit() == ParallelUnit::NotParallel) << "Sparsely accelerating a dense workspace only works within serial loops";


    TensorVar var;
    for (auto it = tensorVars.begin(); it != tensorVars.end(); ++it) {
      if (it->second == locators[0].getTensor() && util::contains(tempToIndexList, it->first)) {
        var = it->first;
        break;
      }
    }

    Expr indexList = tempToIndexList.at(var);
    Expr indexListSize = tempToIndexListSize.at(var);
    Expr bitGuard = tempToBitGuard.at(var);
    Expr loopVar = ir::Var::make(var.getName() + "_index_locator", taco::Int32, false, false);
    Expr coordinate = getCoordinateVar(forall.getIndexVar());

    if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
      markAssignsAtomicDepth++;
      atomicParallelUnit = forall.getParallelUnit();
    }

    Stmt declareVar = VarDecl::make(coordinate, Load::make(indexList, loopVar));
    Stmt body = lowerForallBody(coordinate, forall.getStmt(), locators, inserters, appenders, caseLattice, reducedAccesses);
    Stmt resetGuard = ir::Store::make(bitGuard, coordinate, ir::Literal::make(false), markAssignsAtomicDepth > 0, atomicParallelUnit);

    if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
      markAssignsAtomicDepth--;
    }

    body = Block::make(declareVar, recoveryStmt, body, resetGuard);

    Stmt posAppend = generateAppendPositions(appenders);

    LoopKind kind = LoopKind::Serial;
    if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
      kind = LoopKind::Vectorized;
    }
    else if (forall.getParallelUnit() != ParallelUnit::NotParallel
             && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
      kind = LoopKind::Runtime;
    }

    return Block::blanks(For::make(loopVar, 0, indexListSize, 1, body, kind,
                                         ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(),
                                         ignoreVectorize ? 0 : forall.getUnrollFactor()),
                                         posAppend);
  }

Stmt LowererImpl::lowerForallCoordinate(Forall forall, Iterator iterator,
                                        vector<Iterator> locators,
                                        vector<Iterator> inserters,
                                        vector<Iterator> appenders,
                                        MergeLattice caseLattice,
                                        set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt) {
  // TODO: Support scheduling

  Expr coordinate = iterator.getCoordVar();//getCoordinateVar(forall.getIndexVar());
  Expr position = iterator.getPosVar();
  ModeFunction coordAccess = iterator.coordAccess(parentCoordinates(iterator));

  Stmt declarePosition = Stmt();
  if (provGraph.isCoordVariable(forall.getIndexVar())) {
    if (coordAccess[1].as<ir::Literal>() && coordAccess[1].as<ir::Literal>()->getBoolValue()) {
      declarePosition = VarDecl::make(position, coordAccess[0]);
    } else {
      taco_not_supported_yet; // TODO: We only support accesses that must succeed
    }
  } else {
    taco_not_supported_yet; // TODO: I'm not entirely sure why this check is here
  }

  Stmt coordAccessBlock = Block::blanks(coordAccess.compute(), declarePosition);

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, caseLattice, reducedAccesses);
  body = Block::make(recoveryStmt, body); // TODO: What is this for?

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  auto parentCoords = parentCoordinates(iterator);
  if (!provGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    ModeFunction bounds = iterator.coordBounds(parentCoords); //posBounds(parentPos);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

    taco_not_supported_yet; // TODO: I'm not entirely sure what this is for

//    // E.g. a compressed mode with duplicates. Apply iterator chaining
//    Expr parentSegend = iterator.getParent().getSegendVar();
//    ModeFunction startBounds = iterator.posBounds(parentPos);
//    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
//    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
//    startBound = startBounds[0];
//    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != ParallelUnit::NotParallel
           && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

  // Loop with preamble and postamble
  auto retVal = Block::blanks(boundsCompute,
                              For::make(iterator.getCoordVar(), startBound, endBound, 1,
                                        Block::make(coordAccessBlock, VarDecl::make(getCoordinateVar(forall.getIndexVar()), coordinate), body),
                                        kind,
                                        ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                              posAppend);
  return retVal;
}

Stmt LowererImpl::lowerForallPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders,
                                      MergeLattice caseLattice,
                                      set<Access> reducedAccesses,
                                      ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Stmt declareCoordinate = Stmt();
  Stmt strideGuard = Stmt();
  Stmt boundsGuard = Stmt();
  if (provGraph.isCoordVariable(forall.getIndexVar())) {
    Expr values = getValuesArrayFromIterator(iterator);
    Expr coordinateArray = iterator.posAccess(positions(iterator),
                                              coordinates(iterator), values,
                                              tensorTypes[iterator.getTensor()].getDataType()).getResults()[0];
    // If the iterator is windowed, we must recover the coordinate index
    // variable from the windowed space.
    if (iterator.isWindowed()) {
      if (iterator.isStrided()) {
        // In this case, we're iterating over a compressed level with a for
        // loop. Since the iterator variable will get incremented by the for
        // loop, the guard introduced for stride checking doesn't need to
        // increment the iterator variable.
        strideGuard = this->strideBoundsGuard(iterator, coordinateArray, false /* incrementPosVar */);
      }
      coordinateArray = this->projectWindowedPositionToCanonicalSpace(iterator, coordinateArray);
      // If this forall is being parallelized via CPU threads (OpenMP), then we can't
      // emit a `break` statement, since OpenMP doesn't support breaking out of a
      // parallel loop. Instead, we'll bound the top of the loop and omit the check.
      if (forall.getParallelUnit() != ParallelUnit::CPUThread) {
        boundsGuard = this->upperBoundGuardForWindowPosition(iterator, coordinate);
      }
    }
    declareCoordinate = VarDecl::make(coordinate, coordinateArray);
  }
  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(), locators, inserters, appenders, caseLattice, reducedAccesses);

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  body = Block::make(recoveryStmt, body);

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  Expr parentPos = iterator.getParent().getPosVar();
  if (!provGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    ModeFunction bounds = iterator.posBounds(parentPos);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
    // If we have a window on this iterator, then search for the start of
    // the window rather than starting at the beginning of the level.
    if (iterator.isWindowed()) {
      auto startBoundCopy = startBound;
      startBound = this->searchForStartOfWindowPosition(iterator, startBound, endBound);
      // As discussed above, if this position loop is parallelized over CPU
      // threads (OpenMP), then we need to have an explicit upper bound to
      // the for loop, instead of breaking out of the loop in the middle.
      if (forall.getParallelUnit() == ParallelUnit::CPUThread) {
        endBound = this->searchForEndOfWindowPosition(iterator, startBoundCopy, endBound);
      }
    }
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

    // E.g. a compressed mode with duplicates. Apply iterator chaining
    Expr parentSegend = iterator.getParent().getSegendVar();
    ModeFunction startBounds = iterator.posBounds(parentPos);
    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
    startBound = startBounds[0];
    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != ParallelUnit::NotParallel
           && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

  // Loop with preamble and postamble
  return Block::blanks(
                       boundsCompute,
                       For::make(iterator.getPosVar(), startBound, endBound, 1,
                                 Block::make(strideGuard, declareCoordinate, boundsGuard, body),
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                       posAppend);

}

ir::Expr LowererImpl::getValuesArrayFromIterator(Iterator iterator){
  taco_iassert(util::contains(tensorExprMap, iterator.getTensor()));
  Expr values = getValuesArray(tensorExprMap[iterator.getTensor()]);
  if (iterator.getPosIterKind() == taco_positer_kind::BYTE){
    values = ir::Cast::make(values, UInt8, true);
  }
  return values;
}

Stmt LowererImpl::lowerForallFusedPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders,
                                      MergeLattice caseLattice,
                                      set<Access> reducedAccesses,
                                      ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Stmt declareCoordinate = Stmt();
  if (provGraph.isCoordVariable(forall.getIndexVar())) {
    Expr values = getValuesArrayFromIterator(iterator);
    Expr coordinateArray = iterator.posAccess(positions(iterator),
                                              coordinates(iterator), values,
                                              tensorTypes[iterator.getTensor()].getDataType()).getResults()[0];
    declareCoordinate = VarDecl::make(coordinate, coordinateArray);
  }

  // declare upper-level underived ancestors that will be tracked with while loops
  Expr writeResultCond;
  vector<Stmt> loopsToTrackUnderived;
  vector<Stmt> searchForUnderivedStart;
  std::map<IndexVar, vector<Expr>> coordinateBounds = provGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
  vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
  if (underivedAncestors.size() > 1) {
    // each underived ancestor is initialized to min coordinate bound
    IndexVar posIteratorVar;
#if TACO_ASSERTS
    bool hasIteratorAncestor = provGraph.getPosIteratorAncestor(
        iterator.getIndexVar(), &posIteratorVar);
    taco_iassert(hasIteratorAncestor);
#else /* !TACO_ASSERTS */
    provGraph.getPosIteratorAncestor(
        iterator.getIndexVar(), &posIteratorVar);
#endif /* TACO_ASSERTS */
    // get pos variable then search for leveliterators to find the corresponding iterator

    Iterator posIterator;
    auto iteratorMap = iterators.levelIterators();
    int modePos = -1; // select lowest level possible
    for (auto it = iteratorMap.begin(); it != iteratorMap.end(); it++) {
      if (it->second.getIndexVar() == posIteratorVar && (int) it->first.getModePos() > modePos) {
        posIterator = it->second;
        modePos = (int) it->first.getModePos();
      }
    }
    taco_iassert(posIterator.hasPosIter());

    if (inParallelLoopDepth == 0) {
      for (int i = 0; i < (int) underivedAncestors.size() - 1; i ++) {
        // TODO: only if level is sparse emit underived_pos
        header.push_back(VarDecl::make(this->iterators.modeIterator(underivedAncestors[i]).getPosVar(), 0)); // TODO: set to start position bound
        header.push_back(VarDecl::make(getCoordinateVar(underivedAncestors[i]), coordinateBounds[underivedAncestors[i]][0]));
      }
    } else {
      searchForUnderivedStart.push_back(searchForFusedPositionStart(forall, posIterator));
    }

    Expr parentPos = this->iterators.modeIterator(underivedAncestors[underivedAncestors.size() - 2]).getPosVar();
    ModeFunction posBounds = posIterator.posBounds(parentPos);
    writeResultCond = ir::Eq::make(ir::Add::make(indexVarToExprMap[posIterator.getIndexVar()], 1), posBounds[1]);

    Stmt loopToTrackUnderiveds; // to track next ancestor
    for (int i = 0; i < (int) underivedAncestors.size() - 1; i++) {
      Expr coordVarUnknown = getCoordinateVar(underivedAncestors[i]);
      Expr posVarKnown = this->iterators.modeIterator(underivedAncestors[i+1]).getPosVar();
      if (i == (int) underivedAncestors.size() - 2) {
        posVarKnown = indexVarToExprMap[posIterator.getIndexVar()];
      }
      Expr posVarUnknown = this->iterators.modeIterator(underivedAncestors[i]).getPosVar();

      Iterator posIteratorLevel = posIterator;
      for (int j = (int) underivedAncestors.size() - 2; j > i; j--) { // take parent of iterator enough times to get correct level
        posIteratorLevel = posIteratorLevel.getParent();
      }

      ModeFunction posBoundsLevel = posIteratorLevel.posBounds(posVarUnknown);
      Expr loopcond = ir::Eq::make(posVarKnown, posBoundsLevel[1]);
      Stmt locateCoordVar;
      if (posIteratorLevel.getParent().hasPosIter()) {
        locateCoordVar = ir::Assign::make(coordVarUnknown, ir::Load::make(posIteratorLevel.getParent().getMode().getModePack().getArray(1), posVarUnknown));
      }
      else {
        locateCoordVar = ir::Assign::make(coordVarUnknown, posVarUnknown);
      }
      Stmt loopBody = ir::Block::make(addAssign(posVarUnknown, 1), locateCoordVar, loopToTrackUnderiveds);
      if (posIteratorLevel.getParent().hasPosIter()) { // TODO: if level is unique or not
        loopToTrackUnderiveds = IfThenElse::make(loopcond, loopBody);
      }
      else {
        loopToTrackUnderiveds = While::make(loopcond, loopBody);
      }
    }
    loopsToTrackUnderived.push_back(loopToTrackUnderiveds);
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, caseLattice, reducedAccesses);

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  body = Block::make(recoveryStmt, Block::make(loopsToTrackUnderived), body);

  // Code to write results if using temporary and reset temporary
  if (!whereConsumers.empty() && whereConsumers.back().defined()) {
    Expr temp = tensorVars.find(whereTemps.back())->second;
    Stmt writeResults = Block::make(whereConsumers.back(), ir::Assign::make(temp, ir::Literal::zero(temp.type())));
    body = Block::make(body, IfThenElse::make(writeResultCond, writeResults));
  }

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  if (!provGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    Expr parentPos = iterator.getParent().getPosVar();
    ModeFunction bounds = iterator.posBounds(parentPos);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

    // E.g. a compressed mode with duplicates. Apply iterator chaining
    Expr parentPos = iterator.getParent().getPosVar();
    Expr parentSegend = iterator.getParent().getSegendVar();
    ModeFunction startBounds = iterator.posBounds(parentPos);
    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
    startBound = startBounds[0];
    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != ParallelUnit::NotParallel
           && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }
  // Loop with preamble and postamble
  return Block::blanks(boundsCompute,
                       Block::make(Block::make(searchForUnderivedStart),
                       For::make(indexVarToExprMap[iterator.getIndexVar()], startBound, endBound, 1,
                                 Block::make(declareCoordinate, body),
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor())),
                       posAppend);

}

Stmt LowererImpl::lowerMergeLattice(MergeLattice caseLattice, IndexVar coordinateVar,
                                    IndexStmt statement, 
                                    const std::set<Access>& reducedAccesses)
{
//  std::cout <<  "[lowerMergeLattice]" << std::endl <<
//                "  - Stmt: " << statement << std::endl <<
//                "  - coordinateVar: " << coordinateVar << std::endl <<
//                "  - caseLattice: " << caseLattice << std::endl;

  // Lower merge lattice always gets called from lowerForAll. So we want loop lattice
  MergeLattice loopLattice = caseLattice.getLoopLattice();

  Expr coordinate = getCoordinateVar(coordinateVar);
  vector<Iterator> appenders = filter(loopLattice.results(),
                                      [](Iterator it){return it.hasAppend();});

  vector<Iterator> mergers = loopLattice.points()[0].mergers();
  Stmt iteratorVarInits = codeToInitializeIteratorVars(loopLattice.iterators(), loopLattice.points()[0].rangers(), mergers, coordinate, coordinateVar);

  // if modeiteratornonmerger then will be declared in codeToInitializeIteratorVars
  auto modeIteratorsNonMergers =
          filter(loopLattice.points()[0].iterators(), [mergers](Iterator it){
            bool isMerger = find(mergers.begin(), mergers.end(), it) != mergers.end();
            return it.isDimensionIterator() && !isMerger;
          });
  bool resolvedCoordDeclared = !modeIteratorsNonMergers.empty();

  vector<Stmt> mergeLoopsVec;
  for (MergePoint point : loopLattice.points()) {
    // Each iteration of this loop generates a while loop for one of the merge
    // points in the merge lattice.
    IndexStmt zeroedStmt = zero(statement, getExhaustedAccesses(point, caseLattice));
    MergeLattice sublattice = caseLattice.subLattice(point);
    Stmt mergeLoop = lowerMergePoint(sublattice, coordinate, coordinateVar, zeroedStmt, reducedAccesses, resolvedCoordDeclared, loopLattice.iterators());
//    std::cout << "[lowerMergeLattice] " << sublattice << std::endl << mergeLoop << std::endl;
    mergeLoopsVec.push_back(mergeLoop);
  }
  Stmt mergeLoops = Block::make(mergeLoopsVec);

  // Append position to the pos array
  Stmt appendPositions = generateAppendPositions(appenders);

  return Block::blanks(iteratorVarInits,
                       mergeLoops,
                       appendPositions);
}

bool LowererImpl::isMultiPosIter(const Iterator& it){
  if(it.getTensor().defined() && it.hasPosIter()){
    Expr values = getValuesArrayFromIterator(it);
    auto posAccess = it.posAccess(positions(it), coordinates(it), values,
                                  tensorTypes[it.getTensor()].getDataType());
    if (!posAccess[1].as<ir::Literal>() || !posAccess[1].as<ir::Literal>()->equalsScalar(1)){
      return true;
    }
  }
  return false;
}

Expr ternaryOp(const Expr& c, const Expr& a, const Expr& b){
  // c ? a : b
  Expr a_b = BinOp::make(a,b, " : ");
  return BinOp::make(c, a_b, "(", " ? ", ")");
}

Stmt LowererImpl::lowerMergePoint(MergeLattice pointLattice,
                                  ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                  const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared,
                                  std::vector<Iterator> allIterators)
{
  MergePoint point = pointLattice.points().front();

  vector<Iterator> iterators = point.iterators();
  vector<Iterator> mergers = point.mergers();
  vector<Iterator> rangers = point.rangers();
  vector<Iterator> locators = point.locators();

  taco_iassert(iterators.size() > 0);
  taco_iassert(mergers.size() > 0);
  taco_iassert(rangers.size() > 0);

  auto multiPosIters = filter(iterators, [&](const Iterator& it) { return isMultiPosIter(it); });

  if (!multiPosIters.empty()){
    return lowerMergePointMultiPos(pointLattice, coordinate, coordinateVar, statement,
                                   reducedAccesses, resolvedCoordDeclared, allIterators);
  }


  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates = codeToLoadCoordinatesFromPosIterators(iterators, !resolvedCoordDeclared, coordinate);
  Stmt loadCoordIterPositions = codeToLoadPositionsFromCoordIterators(iterators, !resolvedCoordDeclared);

  // Any iterators with an index set have extra work to do at the header
  // of the merge point.
  std::vector<ir::Stmt> indexSetStmts;
  for (auto& iter : filter(iterators, [](Iterator it) { return it.hasIndexSet(); })) {
    // For each iterator A with an index set B, emit the following code:
    //   setMatch = min(A, B); // Check whether A matches its index set at this point.
    //   if (A == setMatch && B == setMatch) {
    //     // If there was a match, project down the values of the iterators
    //     // to be the position variable of the index set iterator. This has the
    //     // effect of remapping the index of A to be the i'th position of the set.
    //     A_coord = B_pos;
    //     B_coord = B_pos;
    //   } else {
    //     // Advance the iterator and it's index set iterator accordingly if
    //     // there wasn't a match.
    //     A_pos += (A == setMatch);
    //     B_pos += (B == setMatch);
    //     // We must continue so that we only proceed to the rest of the cases in
    //     // the merge if there actually is a point present for A.
    //     continue;
    //   }
    auto setMatch = ir::Var::make("setMatch", Int());
    auto indexSetIter = iter.getIndexSetIterator();
    indexSetStmts.push_back(ir::VarDecl::make(setMatch, ir::Min::make(this->coordinates({iter, indexSetIter}))));
    // Equality checks for each iterator.
    auto iterEq = ir::Eq::make(iter.getCoordVar(), setMatch);
    auto setEq = ir::Eq::make(indexSetIter.getCoordVar(), setMatch);
    // Code to shift down each iterator to the position space of the index set.
    auto shiftDown = ir::Block::make(
      ir::Assign::make(iter.getCoordVar(), indexSetIter.getPosVar()),
      ir::Assign::make(indexSetIter.getCoordVar(), indexSetIter.getPosVar())
    );
    // Code to increment both iterator variables.
    auto incr = ir::Block::make(
      addAssign(iter.getIteratorVar(), ir::Cast::make(Eq::make(iter.getCoordVar(), setMatch), iter.getIteratorVar().type())),
      addAssign(indexSetIter.getIteratorVar(), ir::Cast::make(Eq::make(indexSetIter.getCoordVar(), setMatch), indexSetIter.getIteratorVar().type())),
      ir::Continue::make()
    );
    // Code that uses the defined parts together in the if-then-else.
    indexSetStmts.push_back(ir::IfThenElse::make(ir::And::make(iterEq, setEq), shiftDown, incr));
  }

  // Merge iterator coordinate variables
  Stmt resolvedCoordinate = resolveCoordinate(mergers, coordinate, !resolvedCoordDeclared);

  // UPDATE THE FILLS
  vector<Stmt> fillUpdaters;
  {
    auto updaters = filter(iterators, [](Iterator it){return it.updatesFillRegion();});
    for (auto& it : updaters) {
      Expr values = getValuesArrayFromIterator(it);
      ModeFunction getFill = it.getFillRegion(it.getPosVar(), coordinates(it),
                       values, tensorTypes[it.getTensor()].getDataType());
      fillUpdaters.push_back(getFill.compute());

      auto fillRegion = GetProperty::make(it.getTensor(), TensorProperty::FillRegion);
      auto fillVariable = GetProperty::make(it.getTensor(), TensorProperty::FillValue);

      auto fillRegionReplacement = getFill[0];
      auto length = getFill[1];
      auto updateFill = getFill[2];

      auto fillVars = getFillRegionVars(it.getTensor());
      auto lengthVar = fillVars[0];
      auto indexVar = fillVars[1];

      if (length.as<ir::Literal>() && length.as<ir::Literal>()->equalsScalar(1)){
        auto body = Block::make(Assign::make(fillVariable, Load::make(fillRegionReplacement)));
        fillUpdaters.push_back(IfThenElse::make(updateFill, body));
      } else {
        auto body = Block::make(
                Assign::make(lengthVar, length),
                Assign::make(indexVar, ternaryOp(ir::Eq::make(lengthVar,1), 0, 1)),
                Assign::make(fillRegion, fillRegionReplacement),
                Assign::make(fillVariable, Load::make(fillRegion, 0)));
        fillUpdaters.push_back(IfThenElse::make(updateFill, body));
      }
    }
  }
  Stmt fillUpdatersStmt = fillUpdaters.empty() ? Stmt() : Block::make(fillUpdaters);

  // Locate positions
  Stmt loadLocatorPosVars = declLocatePosVars(locators);

  // Deduplication loops
  auto dupIters = filter(iterators, [](Iterator it){return !it.isUnique() &&
                                                           it.hasPosIter();});
  bool alwaysReduce = (mergers.size() == 1 && mergers[0].hasPosIter());
  Stmt deduplicationLoops = reduceDuplicateCoordinates(coordinate, dupIters,
                                                       alwaysReduce);

  // One case for each child lattice point lp
  Stmt caseStmts = lowerMergeCases(coordinate, coordinateVar, statement, pointLattice,
                                   reducedAccesses, allIterators, iterators);


  // TODO
  Stmt updateFillRegions = codeToUpdateFills(statement, coordinate, coordinateVar, allIterators, mergers);

  // Increment iterator position variables
  Stmt incIteratorVarStmts = codeToIncIteratorVars(coordinate, coordinateVar, iterators, mergers);

  /// While loop over rangers
  return While::make(checkThatNoneAreExhausted(rangers),
                     Block::make(loadPosIterCoordinates,
                                 loadCoordIterPositions,
                                 ir::Block::make(indexSetStmts),
                                 resolvedCoordinate,
                                 fillUpdatersStmt,
                                 loadLocatorPosVars,
                                 deduplicationLoops,
                                 caseStmts,
                                 updateFillRegions,
                                 incIteratorVarStmts));
}

Stmt LowererImpl::codeToSetupCountVars(const vector<Iterator>& iterators){
  vector<Stmt> stmts;

  for (auto& it : iterators){
    ir::Expr var = it.getCountVar();

    Expr values = getValuesArrayFromIterator(it);
    auto posAccess = it.posAccess(positions(it), coordinates(it), values,
                                  tensorTypes[it.getTensor()].getDataType());
    stmts.push_back(VarDecl::make(var, posAccess[1]));
  }

  return Block::make(stmts);
}

Stmt LowererImpl::lowerMergePointMultiPos(MergeLattice pointLattice,
                                          ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                          const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared,
                                          std::vector<Iterator> allIterators)
{
  MergePoint point = pointLattice.points().front();

  vector<Iterator> iterators = point.iterators();
  vector<Iterator> mergers = point.mergers();
  vector<Iterator> rangers = point.rangers();
  vector<Iterator> locators = point.locators();
  auto posIters = filter(iterators, [&](Iterator it) { return it.hasPosIter();});
  auto multiPosIters = filter(posIters, [&](Iterator it) {
      auto posAccess = it.posAccess(positions(it), coordinates(it), getValuesArrayFromIterator(it),
                                    tensorTypes[it.getTensor()].getDataType());
      return !posAccess[1].as<ir::Literal>() || !posAccess[1].as<ir::Literal>()->equalsScalar(1);
  });

  taco_iassert(iterators.size() > 0);
  taco_iassert(mergers.size() > 0);
  taco_iassert(rangers.size() > 0);
  taco_iassert(multiPosIters.size() > 0);

  auto setupCounts = codeToSetupCountVars(posIters);

  vector<Stmt> coordDecls;
  if (!resolvedCoordDeclared){
    for (auto& it : posIters){
      coordDecls.push_back(VarDecl::make(it.getCoordVar(), 0));
    }
  }
  Stmt decls = coordDecls.empty() ? Stmt() : Block::make(coordDecls);

  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates = codeToLoadCoordinatesFromPosIterators(iterators, false, coordinate, true);
  Stmt loadCoordIterPositions = codeToLoadPositionsFromCoordIterators(iterators, !resolvedCoordDeclared);

  // Any iterators with an index set have extra work to do at the header
  // of the merge point.
  std::vector<ir::Stmt> indexSetStmts;
  for (auto& iter : filter(iterators, [](Iterator it) { return it.hasIndexSet(); })) {
    // For each iterator A with an index set B, emit the following code:
    //   setMatch = min(A, B); // Check whether A matches its index set at this point.
    //   if (A == setMatch && B == setMatch) {
    //     // If there was a match, project down the values of the iterators
    //     // to be the position variable of the index set iterator. This has the
    //     // effect of remapping the index of A to be the i'th position of the set.
    //     A_coord = B_pos;
    //     B_coord = B_pos;
    //   } else {
    //     // Advance the iterator and it's index set iterator accordingly if
    //     // there wasn't a match.
    //     A_pos += (A == setMatch);
    //     B_pos += (B == setMatch);
    //     // We must continue so that we only proceed to the rest of the cases in
    //     // the merge if there actually is a point present for A.
    //     continue;
    //   }
    auto setMatch = ir::Var::make("setMatch", Int());
    auto indexSetIter = iter.getIndexSetIterator();
    indexSetStmts.push_back(ir::VarDecl::make(setMatch, ir::Min::make(this->coordinates({iter, indexSetIter}))));
    // Equality checks for each iterator.
    auto iterEq = ir::Eq::make(iter.getCoordVar(), setMatch);
    auto setEq = ir::Eq::make(indexSetIter.getCoordVar(), setMatch);
    // Code to shift down each iterator to the position space of the index set.
    auto shiftDown = ir::Block::make(
            ir::Assign::make(iter.getCoordVar(), indexSetIter.getPosVar()),
            ir::Assign::make(indexSetIter.getCoordVar(), indexSetIter.getPosVar())
    );
    // Code to increment both iterator variables.
    auto incr = ir::Block::make(
            addAssign(iter.getIteratorVar(), ir::Cast::make(Eq::make(iter.getCoordVar(), setMatch), iter.getIteratorVar().type())),
            addAssign(indexSetIter.getIteratorVar(), ir::Cast::make(Eq::make(indexSetIter.getCoordVar(), setMatch), indexSetIter.getIteratorVar().type())),
            ir::Continue::make()
    );
    // Code that uses the defined parts together in the if-then-else.
    indexSetStmts.push_back(ir::IfThenElse::make(ir::And::make(iterEq, setEq), shiftDown, incr));
  }

  // Merge iterator coordinate variables
  Stmt resolvedCoordinate = resolveCoordinate(mergers, coordinate, !resolvedCoordDeclared);

  // Locate positions
  Stmt loadLocatorPosVars = declLocatePosVars(locators);

  // Deduplication loops
  auto dupIters = filter(iterators, [](Iterator it){return !it.isUnique() &&
                                                           it.hasPosIter();});
  bool alwaysReduce = (mergers.size() == 1 && mergers[0].hasPosIter());
  Stmt deduplicationLoops = reduceDuplicateCoordinates(coordinate, dupIters,
                                                       alwaysReduce);

  // One case for each child lattice point lp
  Stmt caseStmts = lowerMergeCases(coordinate, coordinateVar, statement, pointLattice,
                                   reducedAccesses, allIterators, iterators, true);


  // TODO
  Stmt updateFillRegions = codeToUpdateFills(statement, coordinate, coordinateVar, allIterators, mergers);

  // Increment iterator position variables
  Stmt incIteratorVarStmts = codeToIncIteratorVars(coordinate, coordinateVar, iterators, mergers);

  /// While loop over rangers
  return Block::make(setupCounts, decls,
          While::make(checkThatNoneAreExhausted(rangers),
                     Block::make(loadPosIterCoordinates,
                                 loadCoordIterPositions,
                                 ir::Block::make(indexSetStmts),
                                 resolvedCoordinate,
                                 loadLocatorPosVars,
                                 deduplicationLoops,
                                 caseStmts,
                                 updateFillRegions,
                                 incIteratorVarStmts)));
}

Stmt LowererImpl::resolveCoordinate(std::vector<Iterator> mergers, ir::Expr coordinate, bool emitVarDecl) {
  if (mergers.size() == 1) {
    Iterator merger = mergers[0];
    if (merger.hasPosIter()) {
      Expr values = getValuesArrayFromIterator(merger);
      // Just one position iterator so it is the resolved coordinate
      ModeFunction posAccess = merger.posAccess(positions(merger),
                                                coordinates(merger),
                                                values, tensorTypes[merger.getTensor()].getDataType());
      auto access = posAccess[0];
      auto windowVarDecl = Stmt();
      auto stride = Stmt();
      auto guard = Stmt();
      // If the iterator is windowed, we must recover the coordinate index
      // variable from the windowed space.
      if (merger.isWindowed()) {

        // If the iterator is strided, then we have to skip over coordinates
        // that don't match the stride. To do that, we insert a guard on the
        // access. We first extract the access into a temp to avoid emitting
        // a duplicate load on the _crd array.
        if (merger.isStrided()) {
          windowVarDecl = VarDecl::make(merger.getWindowVar(), access);
          access = merger.getWindowVar();
          // Since we're merging values from a compressed array (not iterating over it),
          // we need to advance the outer loop if the current coordinate is not
          // along the desired stride. So, we pass true to the incrementPosVar
          // argument of strideBoundsGuard.
          stride = this->strideBoundsGuard(merger, access, true /* incrementPosVar */);
        }

        access = this->projectWindowedPositionToCanonicalSpace(merger, access);
        guard = this->upperBoundGuardForWindowPosition(merger, coordinate);
      }
      Stmt resolution = emitVarDecl ? VarDecl::make(coordinate, access) : Assign::make(coordinate, access);
      return Block::make(posAccess.compute(),
                         windowVarDecl,
                         stride,
                         resolution,
                         guard);
    }
    else if (merger.hasCoordIter()) {
      ModeFunction coordAccess = merger.coordAccess(parentCoordinates(merger));
      // TODO: This does not check whether the value was found or windowed
      Stmt posResolution = emitVarDecl ? VarDecl::make(merger.getPosVar(), coordAccess[0]) :
                           Assign::make(merger.getPosVar(), coordAccess[0]);

      Stmt coordResolution = emitVarDecl ? VarDecl::make(coordinate, merger.getCoordVar()) :
                             Assign::make(coordinate, merger.getCoordVar());

      return Block::make(coordAccess.compute(),
                         posResolution,
                         coordResolution);
    }
    else if (merger.isDimensionIterator()) {
      // Just one dimension iterator so resolved coordinate already exist and we
      // do nothing
      return Stmt();
    }
    else {
      taco_ierror << "Unexpected type of single iterator " << merger;
      return Stmt();
    }
  }
  else {
    // Multiple position iterators so the smallest is the resolved coordinate
    if (emitVarDecl) {
      return VarDecl::make(coordinate, Min::make(coordinates(mergers)));
    }
    else {
      return Assign::make(coordinate, Min::make(coordinates(mergers)));
    }
  }
}

Expr loadByType(ir::Expr values, ir::Expr pos, Datatype type, bool addr){
  return Load::make(ir::Cast::make(Load::make(std::move(values), std::move(pos), true), type, true), 0, addr);
}


Stmt LowererImpl::applyRLEDenseReduction(std::vector<Iterator> filteredIters, std::vector<Iterator> iterators, 
                                         Stmt body, ir::Expr coordinate, bool& appliedOpt){
  Expr newCoord;
  for (auto& it: filteredIters){
    if (newCoord.defined()) newCoord = ir::Min::make(newCoord, it.getCoordVar());
    else newCoord = it.getCoordVar();
  }

  if (!newCoord.defined()){
    if (iterators.size() == 1 && iterators[0].isFull()) {
      std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(iterators[0].getIndexVar(), definedIndexVarsOrdered,
                                                                underivedBounds, indexVarToExprMap,
                                                                this->iterators);
      newCoord = bounds[1];
    } else {
      return body;
    }
  }
//      std::cout << "newCoord : " << newCoord << std::endl;

  Expr forVar = Var::make("fv", Int());
  Expr accumulator = Var::make("acc", Int());

  const auto* block = body.as<ir::Block>();
  if (block && !block->contents.empty()) {
    const auto *assign = block->contents[0].as<ir::Assign>();
//        std::cout << std::endl << "assign: " << assign << std::endl;
//        std::cout << "body: " << body << std::endl;
    if (assign) {
      const auto *add = assign->rhs.as<ir::Add>();
//          std::cout << "add: " << add << std::endl;
      if (add) {
//            std::cout << "Inner expression within reduction: " << add->b << std::endl;
//            std::cout << "LHS: " << add->a << std::endl;
//            std::cout << "all iters: " << allIterators << std::endl;
//            std::cout << "iters: " << iterators << std::endl;
//            std::cout << "filtered iters: " << filteredIters << std::endl;
        const auto *mul = add->b.as<ir::Mul>();
        if (mul && !mul->a.as<ir::Load>()) {
          const auto * load = mul->b.as<ir::Load>();
          if (load) {
            if (!newCoord.defined()){
              return body;
            }
            Expr vecLoad = ir::Load::make(load->arr, forVar);
            Expr lhs = assign->lhs;

            Stmt forLoop = For::make(forVar, coordinate, newCoord, 1,
                                      Block::make(addAssign(accumulator, vecLoad)));
            appliedOpt = true;
            return Block::make(VarDecl::make(accumulator, 0),
                                forLoop,
                                Assign::make(coordinate, newCoord),
                                addAssign(lhs, ir::Mul::make(accumulator, mul->a)),
                                Continue::make());
          }
        }
      }
    }
  }
  return body;
}

Stmt LowererImpl::lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                  MergeLattice caseLattice,
                                  const std::set<Access>& reducedAccesses,
                                  std::vector<Iterator> allIterators,
                                  std::vector<Iterator> iterators,
                                  bool multiPos)
{
  vector<Stmt> result;

  if (caseLattice.anyModeIteratorIsLeaf() && caseLattice.needExplicitZeroChecks()) {
    // Can check value array of some tensor
    Stmt body = lowerMergeCasesWithExplicitZeroChecks(coordinate, coordinateVar, stmt, caseLattice, reducedAccesses);
    result.push_back(body);
    return Block::make(result);
  }

  // Emitting structural cases so unconditionally apply lattice optimizations.
  MergeLattice loopLattice = caseLattice.getLoopLattice();

  vector<Iterator> appenders;
  vector<Iterator> inserters;
  tie(appenders, inserters) = splitAppenderAndInserters(loopLattice.results());

  auto fillRegionSizeOne = [&](Iterator it){
    Expr values = getValuesArrayFromIterator(it);
    auto fillregion = it.getFillRegion(it.getPosVar(), coordinates(it),
                                       values, tensorTypes[it.getTensor()].getDataType());
    if (!fillregion.defined() || fillregion.getResults().size() == 0) {
      return false;
    }
    auto fill_size = fillregion[1];
    if (fill_size.as<ir::Literal>() != nullptr && fill_size.as<ir::Literal>()->getValue<int>() == 1 ) {
      return true;
    }
    return false;
  };

  bool appliedOpt = false;

  auto fixupBodyReduction = [&](Stmt body, MergePoint point){
    vector<Iterator> rangers = filter(point.rangers(), [&](const Iterator& it){
        return it.getTensor().defined() && it.getIndexVar().defined();
    });

    bool allOpsFill = true;
    bool fillDenseReduction = true;
    bool hasFill = true;
    for (auto& it: allIterators){
      if (!it.updatesFillRegion()){
        allOpsFill = false;
      }
      if (it.updatesFillRegion() && !fillRegionSizeOne(it)){
        std::cout << "Iterator with non-1 fill update: " << it << std::endl;
        hasFill = false;
      }
      if (!it.updatesFillRegion() && it.getTensor().defined() && !it.hasLocate()){
//        std::cout << "fillDenseReduction set to false : " << it << std::endl;
        fillDenseReduction = false;
      }
    }
//    std::cout << "[fixupBodyReduction] allOpsFill : " << allOpsFill << ", hasFill: " << hasFill << ", fillDenseReduction: " << fillDenseReduction << std::endl;
    if (!reducedAccesses.empty() && allOpsFill){
      // Two cases:
      //  - We have a value so we need the next coordinate
      //  - No value, need the current value
      std::vector<Expr> reductionOptExprs;
      for (auto& it : rangers){
        auto p1 = ir::Add::make(it.getPosVar(),1);
        auto access = it.posAccess({p1}, coordinates(it), getValuesArrayFromIterator(it),
                                   tensorTypes[it.getTensor()].getDataType());
        auto ternary = ternaryOp(ir::Lt::make(p1, it.getEndVar()), access[0], GetProperty::make(it.getTensor(), TensorProperty::Dimension, it.getMode().getLevel()-1));
        reductionOptExprs.push_back(ternary);
      }
      vector<Iterator> iterFillRegions = filter(iterators, [&](const Iterator& it){
          return std::find(rangers.begin(), rangers.end(), it) == rangers.end() && it.getTensor().defined();
      });
      for (auto& it : iterFillRegions) {
        reductionOptExprs.push_back(it.getCoordVar());
      }
      ir::Expr mul = ir::Sub::make(ir::Min::make(reductionOptExprs), coordinate);

      const auto* block = body.as<ir::Block>();
      if (block && !block->contents.empty()) {
        const auto *assign = block->contents[0].as<ir::Assign>();
//          std::cout << std::endl << "assign: " << assign << std::endl;
//          std::cout << "body: " << body << std::endl;
        if (assign) {
          const auto *add = assign->rhs.as<ir::Add>();
//            std::cout << "add: " << add << std::endl;
          if (add) {
//            std::cout << "Inner expression within reduction: " << add->b << std::endl;
//            std::cout << "LHS: " << add->a << std::endl;
//            std::cout << "all iters: " << allIterators << std::endl;
            return Block::make(addAssign(add->a, ir::Mul::make(add->b, mul)));
          }
        }
      }
    } else if (!reducedAccesses.empty() && hasFill && fillDenseReduction){
      vector<Iterator> filteredIters = filter(iterators, [&](const Iterator& it){
        return it.getTensor().defined() && it.getIndexVar().defined();
      });
      return applyRLEDenseReduction(filteredIters, iterators, body, coordinate, appliedOpt);
    }
    return body;
  };

  auto fixupBodyReductionDenseZero = [&](Stmt body, MergePoint point){
      vector<Iterator> rangers = filter(point.rangers(), [&](const Iterator& it){
          return it.getTensor().defined() && it.getIndexVar().defined();
      });
      vector<Iterator> filteredAllIters = filter(allIterators, [&](const Iterator& it){
          return it.getTensor().defined() && it.getIndexVar().defined();
      });
      vector<Iterator> filteredIters = filter(iterators, [&](const Iterator& it){
        return it.getTensor().defined() && it.getIndexVar().defined();
      });

    Expr comparison;
      Expr newCoord;
      if (!reducedAccesses.empty() && rangers.empty() && !appliedOpt){
        for (auto& it: filteredIters){
          if (newCoord.defined()) newCoord = ir::Min::make(newCoord, it.getCoordVar());
          else newCoord = it.getCoordVar();
        }
        for (auto& it: filteredAllIters) {
          auto fill = GetProperty::make(it.getTensor(), TensorProperty::FillValue);
          if (it.updatesFillRegion() && !fillRegionSizeOne(it)){
            return body;
          }
          if (comparison.defined()) comparison = ir::Or::make(comparison, ir::Eq::make(fill, 0));
          else comparison = ir::Eq::make(fill, 0);
        }

        Expr forVar = Var::make("fv", Int());
        Stmt elseBlock;
        const auto* block = body.as<ir::Block>();
        if (block && !block->contents.empty()) {
          const auto *assign = block->contents[0].as<ir::Assign>();
          if (assign) {
            const auto *add = assign->rhs.as<ir::Add>();
            if (add) {
              const auto *mul = add->b.as<ir::Mul>();
              if (mul && !mul->a.as<ir::Load>()) {
                const auto * load = mul->b.as<ir::Load>();
                if (load) {
                  if (!newCoord.defined()){
                    return body;
                  }
                  Expr vecLoad = ir::Load::make(load->arr, forVar);
                  Expr lhs = assign->lhs;
                  Stmt newBody = addAssign(lhs, ir::Mul::make(mul->a, Load::make(load->arr, forVar)));

                  elseBlock = Block::make(For::make(forVar, coordinate, newCoord, 1, Block::make(newBody)), 
                                   Assign::make(coordinate, newCoord),
                                   Continue::make());
                }
              }
            }
          }
        }
        if (elseBlock.defined()){
          return IfThenElse::make(comparison, Block::make(Assign::make(coordinate, newCoord), Continue::make()), elseBlock);
        }
      }
//      std::cout << "iterators: " << iterators << std::endl;
//      std::cout << "rangers: " << rangers << std::endl;
//      std::cout << "locators: " << point.locators() << std::endl;
      return body;
  };

  if (loopLattice.iterators().size() == 1) {
    // Just one iterator so no conditional
    taco_iassert(!loopLattice.points()[0].isOmitter());
    Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                appenders, loopLattice, reducedAccesses);
//    std::cout << "[looplattice size 1]" << std::endl << body << std::endl;
    body = fixupBodyReduction(body, loopLattice.points()[0]);
   body = fixupBodyReductionDenseZero(body, loopLattice.points()[0]);
    result.push_back(body);
  }
  else if (!loopLattice.points().empty()) {
    // TODO: HACK
    vector<Expr> coordVars;
    for (Iterator iterator : loopLattice.points()[0].rangers()) {
      if (!(provGraph.isCoordVariable(iterator.getIndexVar()) &&
            provGraph.isDerivedFrom(iterator.getIndexVar(), coordinateVar))) {
        coordVars.push_back(iterator.getCoordVar());
      }
    }

    vector<pair<Expr,Stmt>> cases;
    for (MergePoint point : loopLattice.points()) {
      // Construct case expression
      vector<Expr> coordComparisons = compareToResolvedCoordinate<Eq>(point.rangers(), coordinate, coordinateVar);

      for (auto& ranger : point.rangers()){
        if(ranger.getTensor().defined()) {
          Expr values = getValuesArrayFromIterator(ranger);
          auto posAccess = ranger.posAccess(positions(ranger), coordinates(ranger),
                                            values, tensorTypes[ranger.getTensor()].getDataType());
          // TODO : Should check literal value but it
          if (posAccess[1].as<ir::Literal>() == nullptr) {
            coordComparisons.push_back(posAccess[1]); // Need to check that the coordinate has been found also
          }
        }
      }

      vector<Iterator> omittedRegionIterators = loopLattice.retrieveRegionIteratorsToOmit(point);
      if (!point.isOmitter()) {
        std::vector <Expr> neqComparisons = compareToResolvedCoordinate<Neq>(omittedRegionIterators, coordinate,
                                                                             coordinateVar);
        append(coordComparisons, neqComparisons);
      }

      coordComparisons = filter(coordComparisons, [](const Expr& e) { return e.defined(); });

      // Construct case body
      IndexStmt zeroedStmt = zero(stmt, getExhaustedAccesses(point, loopLattice));
      Stmt body = lowerForallBody(coordinate, zeroedStmt, {},
                                  inserters, appenders, MergeLattice({point}), reducedAccesses);
      body = fixupBodyReduction(body, point);
      body = fixupBodyReductionDenseZero(body, point);


      auto fillsResult = AllFillsVisitor().check(zeroedStmt);
      if (std::get<0>(fillsResult) && appenders.size() == 1 && appenders[0].hasAppendFillRegion()) {
        auto appender = appenders[0];

        if (std::get<1>(fillsResult).size() == 1){
          // We can copy repetitions directly to the output tensor in this case!
          auto lhsType = tensorTypes[allIterators[0].getTensor()].getDataType();
          auto lhsFillRegion = GetProperty::make(allIterators[0].getTensor(), TensorProperty::FillRegion);
          auto lhsValues = getValuesArrayFromIterator(allIterators[0]);
          auto lhsEnd = loadByType(lhsValues, ir::Sub::make(allIterators[0].getPosVar(), 4), lhsType, true );
//          ir::Load::make()

          auto lcmLengths = Var::make("lengthsLcm", Int());
          auto lcmDecl = VarDecl::make(lcmLengths, getFillRegionVars(allIterators[0].getTensor())[0]);
          auto minCoords = Var::make("coordMin", Int());
          auto minDecl = VarDecl::make(minCoords, Min::make(coordVars));
          Expr startVar = Var::make("startVar", Int());
          Stmt startDecl = VarDecl::make(startVar, ir::Sub::make(appender.getPosVar(), ir::Mul::make(ir::Sub::make(lhsEnd, lhsFillRegion), lhsType.getNumBytes())));
          Expr run = Var::make("runValue", Int());
          Stmt runDecl = VarDecl::make(run, ir::Sub::make(minCoords, coordinate));
          Expr values = getValuesArrayFromIterator(appender);

          auto appendFill = appender.getFillRegionAppend(appender.getPosVar(), coordinate,
                                                         startVar, lcmLengths, run,
                                                         values, getCapacityVar(appender.getTensor()),
                                                         tensorTypes[appender.getTensor()].getDataType());
          Stmt coordFixup = Assign::make(coordinate, minCoords);

          body = Block::make(lcmDecl, minDecl, startDecl, runDecl, appendFill, coordFixup, Continue::make());
//                             IfThenElse::make(ifCond, Block::make(whileLoop, Continue::make()),
//                                              Block::make(loopBoundDecl, startDecl, elseLoop, runDecl, appendFill,
//                                                          fillsFixup, incIterators, coordFixup, Continue::make())));

        } else {
          vector<Stmt> whileBodyStmts;
          whileBodyStmts.push_back(body);
          for (auto &iterator : allIterators) {
            if (iterator.updatesFillRegion()) {
              if (checkUpdateFillRequired(iterator)) {
                auto fillVars = getFillRegionVars(iterator.getTensor());
                auto lengthVar = fillVars[0];
                auto indexVar = fillVars[1];
                auto fillRegion = GetProperty::make(iterator.getTensor(), TensorProperty::FillRegion);
                auto fillVariable = GetProperty::make(iterator.getTensor(), TensorProperty::FillValue);

                auto ifLengthOne = IfThenElse::make(ir::Gt::make(lengthVar,1),
                         Block::make(Assign::make(fillVariable, Load::make(fillRegion, indexVar)),
                                     Assign::make(indexVar, ir::Add::make(indexVar, 1)),
                                     IfThenElse::make(ir::Eq::make(indexVar, lengthVar), Assign::make(indexVar, 0))
                                     ));
                whileBodyStmts.push_back(ifLengthOne);
//                whileBodyStmts.push_back(Assign::make(fillVariable, Load::make(fillRegion, indexVar)));
//                whileBodyStmts.push_back(Assign::make(indexVar, ir::Rem::make(ir::Add::make(indexVar, 1), lengthVar)));
              }
            }
          }
          whileBodyStmts.push_back(addAssign(coordinate, 1));

          std::vector<Expr> lengths;
          for (auto &iterator : allIterators) {
            if (iterator.updatesFillRegion()) {
              auto fillVars = getFillRegionVars(iterator.getTensor());
              auto lengthVar = fillVars[0];
              lengths.push_back(lengthVar);
            }
          }
          if (lengths.empty()) {
            lengths.emplace_back(1);
          }

          auto lcmLengths = Var::make("lengthsLcm", Int());
          auto lcmDecl = VarDecl::make(lcmLengths, Lcm::make(lengths));
          auto minCoords = Var::make("coordMin", Int());
          auto minDecl = VarDecl::make(minCoords, Min::make(coordVars));

          Expr startVar = Var::make("startVar", Int());
          Stmt startDecl = VarDecl::make(startVar, appender.getPosVar());
          Expr loopBound = Var::make("loopBound", Int());
          Stmt loopBoundDecl = VarDecl::make(loopBound, ir::Add::make(coordinate, lcmLengths));

          auto whileCond = Lt::make(coordinate, Min::make(minCoords, loopBound));
          auto whileLoop = While::make(whileCond, Block::make(whileBodyStmts));

          auto ifCond = ir::Eq::make(Min::make(minCoords, loopBound), loopBound);

          Expr length = lcmLengths;
          Expr run = Var::make("runValue", Int());
          Stmt runDecl = VarDecl::make(run, ir::Sub::make(minCoords, coordinate));
          Expr values = getValuesArrayFromIterator(appender);
          auto appendFill = appender.getFillRegionAppend(appender.getPosVar(), coordinate,
                                                         startVar, length, run,
                                                         values, getCapacityVar(appender.getTensor()),
                                                         tensorTypes[appender.getTensor()].getDataType());
          Stmt coordFixup = Assign::make(coordinate, minCoords);

          vector<Stmt> fillsFixupVec;
          for (auto &iterator : allIterators) {
            if (iterator.updatesFillRegion()) {
              if (checkUpdateFillRequired(iterator)) {
                auto fillVariable = GetProperty::make(iterator.getTensor(), TensorProperty::FillValue);
                auto fillRegion = GetProperty::make(iterator.getTensor(), TensorProperty::FillRegion);
                auto fillVars = getFillRegionVars(iterator.getTensor());
                auto lengthVar = fillVars[0];
                auto indexVar = fillVars[1];

//                fillsFixupVec.push_back(
//                        Assign::make(indexVar,
//                                     ir::Rem::make(ir::Add::make(indexVar, ir::Sub::make(run, 1)), lengthVar)));
//                fillsFixupVec.push_back(Assign::make(fillVariable, Load::make(fillRegion, indexVar)));
//                fillsFixupVec.push_back(Assign::make(indexVar, ir::Rem::make(ir::Add::make(indexVar, 1), lengthVar)));

                auto ifLengthOne = IfThenElse::make(ir::Gt::make(lengthVar,1),
                                                    Block::make(Assign::make(indexVar, ir::Rem::make(ir::Add::make(indexVar, ir::Sub::make(run, 1)), lengthVar)),
                                                                Assign::make(fillVariable, Load::make(fillRegion, indexVar)),
                                                                Assign::make(indexVar, ir::Add::make(indexVar, 1)),
                                                                IfThenElse::make(ir::Eq::make(indexVar, lengthVar), Assign::make(indexVar, 0))
                                                    ));
                fillsFixupVec.push_back(ifLengthOne);
              }
            }
          }
          Stmt fillsFixup = Block::make(fillsFixupVec);

          vector<Expr> fillsResultTensors;
          for (auto &tensorVar : std::get<1>(fillsResult)) {
            fillsResultTensors.push_back(getTensorVar(tensorVar));
          }

          auto filtered = filter(iterators, [&](Iterator it) {
              return !it.isDimensionIterator();
          });

          // TODO: [DANIELBD] update any other state for multiPos?
          body = Block::make(lcmDecl, minDecl, loopBoundDecl, startDecl, whileLoop,
                             IfThenElse::make(ifCond,
                                 Block::make(runDecl, appendFill,
                                             fillsFixup, coordFixup)),  Continue::make());
        }
      }
      else if (multiPos){
        vector<Iterator> rangers = filter(point.rangers(), [&](const Iterator& it){ return it.getTensor().defined() && it.getIndexVar().defined(); });
        vector<Iterator> multiPosIters = filter(rangers, [&](const Iterator& it){ return isMultiPosIter(it); });
        vector<Iterator> posIters = filter(rangers, [&](const Iterator& it){ return !isMultiPosIter(it); });
        vector<Iterator> iterFillRegions = filter(iterators, [&](const Iterator& it){
            return std::find(rangers.begin(), rangers.end(), it) == rangers.end() && it.getTensor().defined();
        });
        vector<Iterator> fillsLoop = filter(allIterators, [&](const Iterator& it){
            return std::find(rangers.begin(), rangers.end(), it) == rangers.end() && it.getTensor().defined();
        });
        vector<Stmt> stmts;

        if (!posIters.empty()){
          stmts.push_back(body);
          for (auto& it : multiPosIters){
            stmts.push_back(addAssign(it.getCountVar(), -1));
          }
          for (auto &iterator : iterFillRegions) {
            if (iterator.updatesFillRegion()) {
              auto fillVars = getFillRegionVars(iterator.getTensor());
              auto lengthVar = fillVars[0];
              auto indexVar = fillVars[1];
              auto fillRegion = GetProperty::make(iterator.getTensor(), TensorProperty::FillRegion);
              auto fillVariable = GetProperty::make(iterator.getTensor(), TensorProperty::FillValue);

//              stmts.push_back(Assign::make(fillVariable, Load::make(fillRegion, indexVar)));
//              stmts.push_back(Assign::make(indexVar, ir::Rem::make(ir::Add::make(indexVar, 1), lengthVar)));

              auto ifLengthOne = IfThenElse::make(ir::Gt::make(lengthVar,1),
                                                  Block::make(Assign::make(fillVariable, Load::make(fillRegion, indexVar)),
                                                              Assign::make(indexVar, ir::Add::make(indexVar, 1)),
                                                              IfThenElse::make(ir::Eq::make(indexVar, lengthVar), Assign::make(indexVar, 0))
                                                  ));
              stmts.push_back(ifLengthOne);

            }
          }

          body = Block::make(stmts);
        }
        else
        {
          ir::Expr minFound;
          for (auto& it : multiPosIters) {
            minFound = minFound.defined() ? Min::make(minFound, it.getCountVar()) : it.getCountVar();
          }

          for (auto& it : iterFillRegions){
            auto sub = ir::Sub::make(it.getCoordVar(), coordinate);
            minFound = minFound.defined() ? Min::make(minFound, sub) : sub;
          }
          Expr minVar = Var::make("for_end", Int());
          stmts.push_back(VarDecl::make(minVar, minFound));

          Expr forVar = Var::make("l", Int());
          auto getFor = [&](bool rle){
            vector<Stmt> forStmts;

            const auto* block = body.as<ir::Block>();
            if (block && !block->contents.empty()) {
              const auto *assign = block->contents[0].as<ir::Assign>();
              if (assign) {
                const auto *add = assign->rhs.as<ir::Add>();
                if (add) {
                  const auto *mul = add->b.as<ir::Mul>();
                  if (mul && !mul->a.as<ir::Load>()) {
                    const auto * load = mul->b.as<ir::Load>();
                    if (load) {
                      Expr lhs = assign->lhs;
                      body = addAssign(lhs, ir::Mul::make(mul->a, Load::make(load->arr, coordinate)));
                    }
                  }
                }
              }
            }

            forStmts.push_back(body); // TODO: APPLY RLE OPT HERE
            std::cout << "BODY : " << body << std::endl;
            if (!rle) {
              for (auto &iterator : fillsLoop) {
                if (iterator.updatesFillRegion()) {
                  auto fillVars = getFillRegionVars(iterator.getTensor());
                  auto lengthVar = fillVars[0];
                  auto indexVar = fillVars[1];
                  auto fillRegion = GetProperty::make(iterator.getTensor(), TensorProperty::FillRegion);
                  auto fillVariable = GetProperty::make(iterator.getTensor(), TensorProperty::FillValue);

//                  forStmts.push_back(Assign::make(fillVariable, Load::make(fillRegion, indexVar)));
//                  forStmts.push_back(Assign::make(indexVar, ir::Rem::make(ir::Add::make(indexVar, 1), lengthVar)));

                  auto ifLengthOne = IfThenElse::make(ir::Gt::make(lengthVar,1),
                                                      Block::make(Assign::make(fillVariable, Load::make(fillRegion, indexVar)),
                                                                  Assign::make(indexVar, ir::Add::make(indexVar, 1)),
                                                                  IfThenElse::make(ir::Eq::make(indexVar, lengthVar), Assign::make(indexVar, 0))
                                                      ));
                  forStmts.push_back(ifLengthOne);

                }
              }
            }
            for (auto& it : multiPosIters){
              forStmts.push_back(codeToIncIteratorVars(coordinate, coordinateVar, {it}, {}));
            }
            forStmts.push_back(addAssign(coordinate, 1));
            return For::make(forVar, 0, minVar, 1, Block::make(forStmts));
          };
          vector<Expr> lengths;
          for (auto &iterator : fillsLoop) {
            if (iterator.updatesFillRegion()) {
              auto fillVars = getFillRegionVars(iterator.getTensor());
              lengths.push_back(fillVars[0]);
            }
          }
//          stmts.push_back(getFor(false));

          if (lengths.empty()){
            stmts.push_back(getFor(false));
          } else {
            auto rle_len1_for = getFor(true);
            if (fillsLoop.size() == 1){
              auto& iterator = fillsLoop[0];
              auto fillVariable = GetProperty::make(iterator.getTensor(), TensorProperty::FillValue);
              auto crd_incr = addAssign(coordinate, minVar);
              rle_len1_for = IfThenElse::make(ir::Neq::make(fillVariable, 0), Block::make(rle_len1_for), Block::make(crd_incr));
            }
            stmts.push_back(IfThenElse::make(ir::Eq::make(ir::Max::make(lengths), 1), rle_len1_for, getFor(false)));
          }

          for (auto& it : multiPosIters) {
            stmts.push_back(addAssign(it.getCountVar(), ir::Neg::make(minVar)));
            stmts.push_back(addAssign(it.getCoordVar(), minVar));
          }

          stmts.push_back(Continue::make());

          body = Block::make(stmts);
          stmts.clear();
        }
      }

      if (coordComparisons.empty()) {
        Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                    appenders, MergeLattice({point}), reducedAccesses);
        result.push_back(body);
        break;
      }
      cases.emplace_back(taco::ir::conjunction(coordComparisons), body);
    }
    result.push_back(Case::make(cases, loopLattice.exact()));
  }

  return Block::make(result);
}

ir::Expr LowererImpl::constructCheckForAccessZero(Access access) {
  Expr tensorValue = lower(access);
  IndexExpr zeroVal = Literal::zero(tensorValue.type()); //TODO ARRAY Generalize
  return Neq::make(tensorValue, lower(zeroVal));
}

std::vector<Iterator> LowererImpl::getModeIterators(const std::vector<Iterator>& iters) {
  // For now only check mode iterators.
  return filter(iters, [](const Iterator& it){return it.isModeIterator();});
}

std::vector<ir::Stmt> LowererImpl::constructInnerLoopCasePreamble(ir::Expr coordinate, IndexVar coordinateVar,
                                                                  MergeLattice lattice,
                                                                  map<Iterator, Expr>& iteratorToConditionMap) {
  vector<Stmt> result;

  // First, get mode iterator coordinate comparisons
  std::vector<Iterator> modeIterators = getModeIterators(lattice.iterators());
  vector<Expr> coordComparisonsForModeIters = compareToResolvedCoordinate<Eq>(modeIterators, coordinate, coordinateVar);

  std::vector<Iterator> modeItersWithIndexCases;
  std::vector<Expr> coordComparisons;
  for(size_t i = 0; i < coordComparisonsForModeIters.size(); ++i) {
    Expr expr = coordComparisonsForModeIters[i];
    if (expr.defined()) {
      modeItersWithIndexCases.push_back(modeIterators[i]);
      coordComparisons.push_back(expr);
    }
  }

  // Construct tensor iterators with modeIterators first then locate iterators to keep a mapping between vector indices
  vector<Iterator> tensorIterators = combine(modeItersWithIndexCases, lattice.locators());
  tensorIterators = getModeIterators(tensorIterators);

  // Get value comparisons for all tensor iterators
  vector<Access> itAccesses;
  vector<Expr> valueComparisons;
  for(auto it : tensorIterators) {
    Access itAccess = iterators.modeAccess(it).getAccess();
    itAccesses.push_back(itAccess);
    if(it.isLeaf()) {
      valueComparisons.push_back(constructCheckForAccessZero(itAccess));
    } else {
      valueComparisons.push_back(Expr());
    }
  }

  // Construct isNonZero cases
  for(size_t i = 0; i < coordComparisons.size(); ++i) {
    Expr nonZeroCase;
    if(coordComparisons[i].defined() && valueComparisons[i].defined()) {
      nonZeroCase = conjunction({coordComparisons[i], valueComparisons[i]});
    } else if (valueComparisons[i].defined()) {
      nonZeroCase = valueComparisons[i];
    } else if (coordComparisons[i].defined()) {
      nonZeroCase = coordComparisons[i];
    } else {
      continue;
    }
    Expr caseName = Var::make(itAccesses[i].getTensorVar().getName() + "_isNonZero", taco::Bool);
    Stmt declaration = VarDecl::make(caseName, nonZeroCase);
    result.push_back(declaration);
    iteratorToConditionMap[tensorIterators[i]] = caseName;
  }

  for(size_t i = modeItersWithIndexCases.size(); i < valueComparisons.size(); ++i) {
    Expr caseName = Var::make(itAccesses[i].getTensorVar().getName() + "_isNonZero", taco::Bool);
    Stmt declaration = VarDecl::make(caseName, valueComparisons[i]);
    result.push_back(declaration);
    iteratorToConditionMap[tensorIterators[i]] = caseName;
  }

  return result;
}

vector<Stmt> LowererImpl::lowerCasesFromMap(map<Iterator, Expr> iteratorToCondition,
                                            ir::Expr coordinate, IndexStmt stmt, const MergeLattice& lattice,
                                            const std::set<Access>& reducedAccesses) {

  vector<Iterator> appenders;
  vector<Iterator> inserters;
  tie(appenders, inserters) = splitAppenderAndInserters(lattice.results());

  std::vector<Stmt> result;
  vector<pair<Expr,Stmt>> cases;
  for (MergePoint point : lattice.points()) {

    if(point.isOmitter()) {
      continue;
    }

    // Construct case expression
    vector<Expr> isNonZeroComparisions;
    for(auto& it : combine(point.rangers(), point.locators())) {
      if(util::contains(iteratorToCondition, it)) {
        taco_iassert(iteratorToCondition.at(it).type() == taco::Bool) << "Map must have boolean types";
        isNonZeroComparisions.push_back(iteratorToCondition.at(it));
      }
    }

    function<Expr(const Iterator& it)> getNegatedComparison = [&](const Iterator& it) {return ir::Neg::make(iteratorToCondition.at(it));};
    vector<Iterator> omittedRegionIterators = lattice.retrieveRegionIteratorsToOmit(point);
    for(auto& it : omittedRegionIterators) {
      if(util::contains(iteratorToCondition, it)) {
        isNonZeroComparisions.push_back(ir::Neg::make(iteratorToCondition.at(it)));
      }
    }

    // Construct case body
    IndexStmt zeroedStmt = zero(stmt, getExhaustedAccesses(point, lattice));
    Stmt body = lowerForallBody(coordinate, zeroedStmt, {},
                                inserters, appenders, MergeLattice({point}), reducedAccesses);
    if (isNonZeroComparisions.empty()) {
      Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                  appenders, MergeLattice({point}), reducedAccesses);
      result.push_back(body);
      break;
    }
    cases.push_back({taco::ir::conjunction(isNonZeroComparisions), body});
  }

  vector<Iterator> inputs = combine(lattice.iterators(), lattice.locators());
  inputs = getModeIterators(inputs);

  if(!lattice.exact() && util::any(inserters, [](Iterator it){return it.isFull();}) && hasNoForAlls(stmt)
     && any(inputs, [](Iterator it){return it.isFull();})) {
    // Currently, if the lattice is not exact, the output is full and any of the inputs are full, we initialize
    // the result tensor
    vector<Stmt> stmts;
    for(auto& it : inserters) {
      if(it.isFull()) {
        Access access = iterators.modeAccess(it).getAccess();
        IndexStmt initStmt = Assignment(access, Literal::zero(access.getDataType()));
        Stmt initialization = lowerForallBody(coordinate, initStmt, {}, inserters,
                                              appenders, MergeLattice({}), reducedAccesses);
        stmts.push_back(initialization);
      }
    }
    Stmt backgroundInit = Block::make(stmts);
    cases.push_back({Expr((bool) true), backgroundInit});
    result.push_back(Case::make(cases, true));
  } else {
    result.push_back(Case::make(cases, lattice.exact()));
  }
  return result;
}

/// Lowers a merge lattice to cases assuming there are no more loops to be emitted in stmt.
Stmt LowererImpl::lowerMergeCasesWithExplicitZeroChecks(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                                        MergeLattice lattice, const std::set<Access>& reducedAccesses) {

    vector<Stmt> result;
    if (lattice.points().size() == 1 && lattice.iterators().size() == 1) {
      // Just one iterator so no conditional
      vector<Iterator> appenders;
      vector<Iterator> inserters;
      tie(appenders, inserters) = splitAppenderAndInserters(lattice.results());
      taco_iassert(!lattice.points()[0].isOmitter());
      Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                  appenders, lattice, reducedAccesses);
      result.push_back(body);
    } else if (!lattice.points().empty()) {
      map<Iterator, Expr> iteratorToConditionMap;

      vector<Stmt> preamble = constructInnerLoopCasePreamble(coordinate, coordinateVar, lattice, iteratorToConditionMap);
      util::append(result, preamble);
      vector<Stmt> cases = lowerCasesFromMap(iteratorToConditionMap, coordinate, stmt, lattice, reducedAccesses);
      util::append(result, cases);
    }

    return Block::make(result);
}

Stmt LowererImpl::lowerForallBody(Expr coordinate, IndexStmt stmt,
                                  vector<Iterator> locators,
                                  vector<Iterator> inserters,
                                  vector<Iterator> appenders,
                                  MergeLattice caseLattice,
                                  const set<Access>& reducedAccesses) {

  // Inserter positions
  Stmt declInserterPosVars = declLocatePosVars(inserters);

  // Locate positions
  Stmt declLocatorPosVars = declLocatePosVars(locators);

  if (captureNextLocatePos) {
    capturedLocatePos = Block::make(declInserterPosVars, declLocatorPosVars);
    captureNextLocatePos = false;
  }

  if (caseLattice.anyModeIteratorIsLeaf() && caseLattice.points().size() > 1) {

    // Code of loop body statement
    // Explicit zero checks needed
    std::vector<Stmt> stmts;

    // Need to emit checks based on case lattice
    vector<Iterator> modeIterators = getModeIterators(combine(caseLattice.iterators(), caseLattice.locators()));
    std::map<Iterator, Expr> caseMap;
    for(auto it : modeIterators) {
      if(it.isLeaf()) {
        // Only emit explicit 0 checks for leaf iterators since these are the only iterators can can access tensor
        // values array
        Access itAccess = iterators.modeAccess(it).getAccess();
        Expr accessCase = constructCheckForAccessZero(itAccess);
        caseMap.insert({it, accessCase});
      }
    }

    // This will lower the body for each case to actually compute. Therefore, we don't need to resize assembly arrays
    std::vector<Stmt> loweredCases = lowerCasesFromMap(caseMap, coordinate, stmt, caseLattice, reducedAccesses);

    append(stmts, loweredCases);
    Stmt body = Block::make(stmts);

    return Block::make(declInserterPosVars, declLocatorPosVars, body);
  }

  Stmt initVals = resizeAndInitValues(appenders, reducedAccesses);

  // Code of loop body statement
  Stmt body = lower(stmt);

  // Code to append coordinates
  Stmt appendCoords = appendCoordinate(appenders, coordinate);

  // TODO: Emit code to insert coordinates

  return Block::make(initVals,
                     declInserterPosVars,
                     declLocatorPosVars,
                     body,
                     appendCoords);
}

Expr LowererImpl::getTemporarySize(Where where) {
  TensorVar temporary = where.getTemporary();
  Dimension temporarySize = temporary.getType().getShape().getDimension(0);
  Access temporaryAccess = getResultAccesses(where.getProducer()).first[0];
  std::vector<IndexVar> indexVars = temporaryAccess.getIndexVars();

  if(util::all(indexVars, [&](const IndexVar& var) { return provGraph.isUnderived(var);})) {
    // All index vars underived then use tensor properties to get tensor size
    taco_iassert(util::contains(dimensions, indexVars[0])) << "Missing " << indexVars[0];
    ir::Expr size = dimensions.at(indexVars[0]);
    for(size_t i = 1; i < indexVars.size(); ++i) {
      taco_iassert(util::contains(dimensions, indexVars[i])) << "Missing " << indexVars[i];
      size = ir::Mul::make(size, dimensions.at(indexVars[i]));
    }
    return size;
  }

  if (temporarySize.isFixed()) {
    return ir::Literal::make(temporarySize.getSize());
  }

  if (temporarySize.isIndexVarSized()) {
    IndexVar var = temporarySize.getIndexVarSize();
    vector<Expr> bounds = provGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds,
                                                     indexVarToExprMap, iterators);
    return ir::Sub::make(bounds[1], bounds[0]);
  }

  taco_ierror; // TODO
  return Expr();
}


vector<Stmt> LowererImpl::codeToInitializeDenseAcceleratorArrays(Where where, bool parallel) {
  // if parallel == true, need to initialize dense accelerator arrays as size*numThreads
  // and rename all dense accelerator arrays to name + '_all'

  TensorVar temporary = where.getTemporary();

  // TODO: emit as uint64 and manually emit bit pack code
  const Datatype bitGuardType = taco::Bool;
  std::string bitGuardSuffix;
  if (parallel)
    bitGuardSuffix = "_already_set_all";
  else
    bitGuardSuffix = "_already_set";
  const std::string bitGuardName = temporary.getName() + bitGuardSuffix;

  Expr bitGuardSize = getTemporarySize(where);
  Expr maxThreads = ir::Call::make("omp_get_max_threads", {}, bitGuardSize.type());
  if (parallel)
    bitGuardSize = ir::Mul::make(bitGuardSize, maxThreads);

  const Expr alreadySetArr = ir::Var::make(bitGuardName,
                                           bitGuardType,
                                           true, false);

  // TODO: TACO should probably keep state on if it can use int32 or if it should switch to
  //       using int64 for indices. This assumption is made in other places of taco.
  const Datatype indexListType = taco::Int32;
  std::string indexListSuffix;
  if (parallel)
    indexListSuffix = "_index_list_all";
  else
    indexListSuffix = "_index_list";

  const std::string indexListName = temporary.getName() + indexListSuffix;
  const Expr indexListArr = ir::Var::make(indexListName,
                                          indexListType,
                                          true, false);

  // no decl for shared memory
  Stmt alreadySetDecl = Stmt();
  Stmt indexListDecl = Stmt();
  Stmt freeTemps = Block::make(Free::make(indexListArr), Free::make(alreadySetArr));
  if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0) || !should_use_CUDA_codegen()) {
    alreadySetDecl = VarDecl::make(alreadySetArr, ir::Literal::make(0));
    indexListDecl = VarDecl::make(indexListArr, ir::Literal::make(0));
  }

  if (parallel) {
    whereToIndexListAll[where] = indexListArr;
    whereToBitGuardAll[where] = alreadySetArr;
  } else {
    const Expr indexListSizeExpr = ir::Var::make(indexListName + "_size", taco::Int32, false, false);
    tempToIndexList[temporary] = indexListArr;
    tempToIndexListSize[temporary] = indexListSizeExpr;
    tempToBitGuard[temporary] = alreadySetArr;
  }

  Stmt allocateIndexList = Allocate::make(indexListArr, bitGuardSize);
  if(should_use_CUDA_codegen()) {
    Stmt allocateAlreadySet = Allocate::make(alreadySetArr, bitGuardSize);
    Expr p = Var::make("p" + temporary.getName(), Int());
    Stmt guardZeroInit = Store::make(alreadySetArr, p, ir::Literal::zero(bitGuardType));

    Stmt zeroInitLoop = For::make(p, 0, bitGuardSize, 1, guardZeroInit, LoopKind::Serial);
    Stmt inits = Block::make(alreadySetDecl, indexListDecl, allocateAlreadySet, allocateIndexList, zeroInitLoop);
    return {inits, freeTemps};
  } else {
    Expr sizeOfElt = Sizeof::make(bitGuardType);
    Expr callocAlreadySet = ir::Call::make("calloc", {bitGuardSize, sizeOfElt}, Int());
    Stmt allocateAlreadySet = VarDecl::make(alreadySetArr, callocAlreadySet);
    Stmt inits = Block::make(indexListDecl, allocateIndexList, allocateAlreadySet);
    return {inits, freeTemps};
  }

}

// Returns true if the following conditions are met:
// 1) The temporary is a dense vector
// 2) There is only one value on the right hand side of the consumer
//    -- We would need to handle sparse acceleration in the merge lattices for
//       multiple operands on the RHS
// 3) The left hand side of the where consumer is sparse, if the consumer is an
//    assignment
// 4) CPU Code is being generated (TEMPORARY - This should be removed)
//    -- The sorting calls and calloc call in lower where are CPU specific. We
//       could map calloc to a cudaMalloc and use a library like CUB to emit
//       the sort. CUB support is built into CUDA 11 but not prior versions of
//       CUDA so in that case, we'd probably need to include the CUB headers in
//       the generated code.
std::pair<bool,bool> LowererImpl::canAccelerateDenseTemp(Where where) {
  // TODO: TEMPORARY -- Needs to be removed
  if(should_use_CUDA_codegen()) {
    return std::make_pair(false, false);
  }

  TensorVar temporary = where.getTemporary();
  // (1) Temporary is dense vector
  if(!isDense(temporary.getFormat()) || temporary.getOrder() != 1) {
    return std::make_pair(false, false);
  }

  // (2) Multiple operands in inputs (need lattice to reason about iteration)
  const auto inputAccesses = getArgumentAccesses(where.getConsumer());
  if(inputAccesses.size() > 1 || inputAccesses.empty()) {
    return std::make_pair(false, false);
  }

  // No or multiple results?
  const auto resultAccesses = getResultAccesses(where.getConsumer()).first;
  if(resultAccesses.size() > 1 || resultAccesses.empty()) {
    return std::make_pair(false, false);
  }

  // No check for size of tempVar since we enforced the temporary is a vector
  // and if there is only one RHS value, it must (should?) be the temporary
  std::vector<IndexVar> tempVar = inputAccesses[0].getIndexVars();

  // Get index vars in result.
  std::vector<IndexVar> resultVars = resultAccesses[0].getIndexVars();
  auto it = std::find_if(resultVars.begin(), resultVars.end(),
      [&](const auto& resultVar) {
          return resultVar == tempVar[0] ||
                 provGraph.isDerivedFrom(tempVar[0], resultVar);
  });

  if (it == resultVars.end()) {
    return std::make_pair(true, false);
  }

  int index = (int)(it - resultVars.begin());
  TensorVar resultTensor = resultAccesses[0].getTensorVar();
  int modeIndex = resultTensor.getFormat().getModeOrdering()[index];
  ModeFormat varFmt = resultTensor.getFormat().getModeFormats()[modeIndex];
  // (3) Level of result is sparse
  if(varFmt.isFull()) {
    return std::make_pair(false, false);
  }

  // Only need to sort the workspace if the result needs to be ordered
  return std::make_pair(true, varFmt.isOrdered());
}

// Code to initialize the local temporary workspace from the shared workspace
// in codeToInitializeTemporaryParallel for a SINGLE parallel unit
// (e.g.) the local workspace that each thread uses
vector<Stmt> LowererImpl::codeToInitializeLocalTemporaryParallel(Where where, ParallelUnit parallelUnit) {
  TensorVar temporary = where.getTemporary();
  vector<Stmt> decls;

  Expr tempSize = getTemporarySize(where);
  Expr threadNum = ir::Call::make("omp_get_thread_num", {}, tempSize.type());
  tempSize = ir::Mul::make(tempSize, threadNum);

  bool accelerateDense = canAccelerateDenseTemp(where).first;

  Expr values;
  if (util::contains(needCompute, temporary) &&
      needComputeValues(where, temporary)) {
    // Declare local temporary workspace array
    values = ir::Var::make(temporary.getName(),
                           temporary.getType().getDataType(),
                           true, false);
    Expr values_all = this->temporaryArrays[this->whereToTemporaryVar[where]].values;
    Expr tempRhs = ir::Add::make(values_all, tempSize);
    Stmt tempDecl = ir::VarDecl::make(values, tempRhs);
    decls.push_back(tempDecl);
  }
  /// Make a struct object that lowerAssignment and lowerAccess can read
  /// temporary value arrays from.
  TemporaryArrays arrays;
  arrays.values = values;
  this->temporaryArrays.insert({temporary, arrays});

  if (accelerateDense) {
    // Declare local index list array
    // TODO: TACO should probably keep state on if it can use int32 or if it should switch to
    //       using int64 for indices. This assumption is made in other places of taco.
    const Datatype indexListType = taco::Int32;
    const std::string indexListName = temporary.getName() + "_index_list";
    const Expr indexListArr = ir::Var::make(indexListName,
                                            indexListType,
                                            true, false);

    Expr indexList_all = this->whereToIndexListAll[where];
    Expr indexListRhs = ir::Add::make(indexList_all, tempSize);
    Stmt indexListDecl = ir::VarDecl::make(indexListArr, indexListRhs);
    decls.push_back(indexListDecl);

    // Declare local indexList size variable
    const Expr indexListSizeExpr = ir::Var::make(indexListName + "_size", taco::Int32, false, false);

    // Declare local already set array (bit guard)
    // TODO: emit as uint64 and manually emit bit pack code
    const Datatype bitGuardType = taco::Bool;
    const std::string bitGuardName = temporary.getName() + "_already_set";
    const Expr alreadySetArr = ir::Var::make(bitGuardName,
                                             bitGuardType,
                                             true, false);
    Expr bitGuard_all = this->whereToBitGuardAll[where];
    Expr bitGuardRhs = ir::Add::make(bitGuard_all, tempSize);
    Stmt bitGuardDecl = ir::VarDecl::make(alreadySetArr, bitGuardRhs);
    decls.push_back(bitGuardDecl);

    tempToIndexList[temporary] = indexListArr;
    tempToIndexListSize[temporary] = indexListSizeExpr;
    tempToBitGuard[temporary] = alreadySetArr;
  }
  return decls;
}

// Code to initialize a temporary workspace that is SHARED across ALL parallel units.
// New temporaries are denoted by temporary.getName() + '_all'
// Currently only supports CPUThreads
vector<Stmt> LowererImpl::codeToInitializeTemporaryParallel(Where where, ParallelUnit parallelUnit) {
  TensorVar temporary = where.getTemporary();
  // For the parallel case, need to hoist up a workspace shared by all threads
  TensorVar temporaryAll = TensorVar(temporary.getName() + "_all", temporary.getType(), temporary.getFormat());
  this->whereToTemporaryVar[where] = temporaryAll;

  bool accelerateDense = canAccelerateDenseTemp(where).first;

  Stmt freeTemporary = Stmt();
  Stmt initializeTemporary = Stmt();

  // When emitting code to accelerate dense workspaces with sparse iteration, we need the following arrays
  // to construct the result indices
  if(accelerateDense) {
    vector<Stmt> initAndFree = codeToInitializeDenseAcceleratorArrays(where, true);
    initializeTemporary = initAndFree[0];
    freeTemporary = initAndFree[1];
  }

  Expr values;
  if (util::contains(needCompute, temporary) &&
      needComputeValues(where, temporary)) {
    values = ir::Var::make(temporaryAll.getName(),
                           temporaryAll.getType().getDataType(),
                                true, false);
    taco_iassert(temporaryAll.getType().getOrder() == 1) << " Temporary order was "
                                                      << temporaryAll.getType().getOrder();  // TODO
    Expr size = getTemporarySize(where);
    Expr sizeAll = ir::Mul::make(size, ir::Call::make("omp_get_max_threads", {}, size.type()));

    // no decl needed for shared memory
    Stmt decl = Stmt();
    if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0) || !should_use_CUDA_codegen()) {
      decl = VarDecl::make(values, ir::Literal::make(0));
    }
    Stmt allocate = Allocate::make(values, sizeAll);

    freeTemporary = Block::make(freeTemporary, Free::make(values));
    initializeTemporary = Block::make(decl, initializeTemporary, allocate);
  }
  /// Make a struct object that lowerAssignment and lowerAccess can read
  /// temporary value arrays from.
  TemporaryArrays arrays;
  arrays.values = values;
  this->temporaryArrays.insert({temporaryAll, arrays});

  return {initializeTemporary, freeTemporary};
}

vector<Stmt> LowererImpl::codeToInitializeTemporary(Where where) {
  TensorVar temporary = where.getTemporary();

  const bool accelerateDense = canAccelerateDenseTemp(where).first;

  Stmt freeTemporary = Stmt();
  Stmt initializeTemporary = Stmt();
  if (isScalar(temporary.getType())) {
    initializeTemporary = defineScalarVariable(temporary, true);
    Expr tempSet = ir::Var::make(temporary.getName() + "_set", Datatype::Bool);
    Stmt initTempSet = VarDecl::make(tempSet, false);
    initializeTemporary = Block::make(initializeTemporary, initTempSet);
    tempToBitGuard[temporary] = tempSet;
  } else {
    // TODO: Need to support keeping track of initialized elements for
    //       temporaries that don't have sparse accelerator
    taco_iassert(!util::contains(guardedTemps, temporary) || accelerateDense);

    // When emitting code to accelerate dense workspaces with sparse iteration, we need the following arrays
    // to construct the result indices
    if(accelerateDense) {
      vector<Stmt> initAndFree = codeToInitializeDenseAcceleratorArrays(where);
      initializeTemporary = initAndFree[0];
      freeTemporary = initAndFree[1];
    }

    Expr values;
    if (util::contains(needCompute, temporary) &&
        needComputeValues(where, temporary)) {
      values = ir::Var::make(temporary.getName(),
                             temporary.getType().getDataType(), true, false);
      taco_iassert(temporary.getType().getOrder() == 1)
          << " Temporary order was " << temporary.getType().getOrder();  // TODO
      Expr size = getTemporarySize(where);

      // no decl needed for shared memory
      Stmt decl = Stmt();
      if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0) || !should_use_CUDA_codegen()) {
        decl = VarDecl::make(values, ir::Literal::make(0));
      }
      Stmt allocate = Allocate::make(values, size);

      freeTemporary = Block::make(freeTemporary, Free::make(values));
      initializeTemporary = Block::make(decl, initializeTemporary, allocate);
    }

    /// Make a struct object that lowerAssignment and lowerAccess can read
    /// temporary value arrays from.
    TemporaryArrays arrays;
    arrays.values = values;
    this->temporaryArrays.insert({temporary, arrays});
  }
  return {initializeTemporary, freeTemporary};
}

Stmt LowererImpl::lowerWhere(Where where) {
  TensorVar temporary = where.getTemporary();
  bool accelerateDenseWorkSpace, sortAccelerator;
  std::tie(accelerateDenseWorkSpace, sortAccelerator) =
      canAccelerateDenseTemp(where);

  // Declare and initialize the where statement's temporary
  vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
  bool temporaryHoisted = false;
  for (auto it = temporaryInitialization.begin(); it != temporaryInitialization.end(); ++it) {
    if (it->second == where && it->first.getParallelUnit() == ParallelUnit::NotParallel && !isScalar(temporary.getType())) {
      temporaryHoisted = true;
    } else if (it->second == where && it->first.getParallelUnit() == ParallelUnit::CPUThread && !isScalar(temporary.getType())) {
      temporaryHoisted = true;
      auto decls = codeToInitializeLocalTemporaryParallel(where, it->first.getParallelUnit());

      temporaryValuesInitFree[0] = ir::Block::make(decls);
    }
  }

  if (!temporaryHoisted) {
    temporaryValuesInitFree = codeToInitializeTemporary(where);
  }

  Stmt initializeTemporary = temporaryValuesInitFree[0];
  Stmt freeTemporary = temporaryValuesInitFree[1];

  match(where.getConsumer(),
        std::function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
            if (op->lhs.getTensorVar().getOrder() > 0) {
              whereTempsToResult[where.getTemporary()] = (const AccessNode *) op->lhs.ptr;
            }
        })
  );

  Stmt consumer = lower(where.getConsumer());
  if (accelerateDenseWorkSpace && sortAccelerator) {
    // We need to sort the indices array
    Expr listOfIndices = tempToIndexList.at(temporary);
    Expr listOfIndicesSize = tempToIndexListSize.at(temporary);
    Expr sizeOfElt = ir::Sizeof::make(listOfIndices.type());
    Stmt sortCall = ir::Sort::make({listOfIndices, listOfIndicesSize, sizeOfElt});
    consumer = Block::make(sortCall, consumer);
  }

  // Now that temporary allocations are hoisted, we always need to emit an initialization loop before entering the
  // producer but only if there is no dense acceleration
  if (util::contains(needCompute, temporary) && !isScalar(temporary.getType()) && !accelerateDenseWorkSpace) {
    // TODO: We only actually need to do this if:
    //      1) We use the temporary multiple times
    //      2) The PRODUCER RHS is sparse(not full). (Guarantees that old values are overwritten before consuming)

    Expr p = Var::make("p" + temporary.getName(), Int());
    Expr values = ir::Var::make(temporary.getName(),
                                temporary.getType().getDataType(),
                                true, false);
    Expr size = getTemporarySize(where);
    Stmt zeroInit = Store::make(values, p, ir::Literal::zero(temporary.getType().getDataType()));
    Stmt loopInit = For::make(p, 0, size, 1, zeroInit, LoopKind::Serial);
    initializeTemporary = Block::make(initializeTemporary, loopInit);
  }

  whereConsumers.push_back(consumer);
  whereTemps.push_back(where.getTemporary());
  captureNextLocatePos = true;

  // don't apply atomics to producer TODO: mark specific assignments as atomic
  bool restoreAtomicDepth = false;
  if (markAssignsAtomicDepth > 0) {
    markAssignsAtomicDepth--;
    restoreAtomicDepth = true;
  }

  Stmt producer = lower(where.getProducer());
  if (accelerateDenseWorkSpace) {
    const Expr indexListSizeExpr = tempToIndexListSize.at(temporary);
    const Stmt indexListSizeDecl = VarDecl::make(indexListSizeExpr, ir::Literal::make(0));
    initializeTemporary = Block::make(indexListSizeDecl, initializeTemporary);
  }

  if (restoreAtomicDepth) {
    markAssignsAtomicDepth++;
  }

  whereConsumers.pop_back();
  whereTemps.pop_back();
  whereTempsToResult.erase(where.getTemporary());
  return Block::make(initializeTemporary, producer, markAssignsAtomicDepth > 0 ? capturedLocatePos : ir::Stmt(), consumer,  freeTemporary);
}


Stmt LowererImpl::lowerSequence(Sequence sequence) {
  Stmt definition = lower(sequence.getDefinition());
  Stmt mutation = lower(sequence.getMutation());
  return Block::make(definition, mutation);
}


Stmt LowererImpl::lowerAssemble(Assemble assemble) {
  Stmt queries, freeQueryResults;
  if (generateAssembleCode() && assemble.getQueries().defined()) {
    std::vector<Stmt> allocStmts, freeStmts;
    const auto queryAccesses = getResultAccesses(assemble.getQueries()).first;
    for (const auto& queryAccess : queryAccesses) {
      const auto queryResult = queryAccess.getTensorVar();
      Expr values = ir::Var::make(queryResult.getName(),
                                  queryResult.getType().getDataType(),
                                  true, false);

      TemporaryArrays arrays;
      arrays.values = values;
      this->temporaryArrays.insert({queryResult, arrays});

      // Compute size of query result
      const auto indexVars = queryAccess.getIndexVars();
      taco_iassert(util::all(indexVars,
          [&](const auto& var) { return provGraph.isUnderived(var); }));
      Expr size = 1;
      for (const auto& indexVar : indexVars) {
        size = ir::Mul::make(size, getDimension(indexVar));
      }

      multimap<IndexVar, Iterator> readIterators;
      for (auto& read : getArgumentAccesses(assemble.getQueries())) {
        for (auto& readIterator : getIterators(read)) {
          for (auto& underivedAncestor :
              provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
            readIterators.insert({underivedAncestor, readIterator});
          }
        }
      }
      const auto writeIterators = getIterators(queryAccess);
      const bool zeroInit = hasSparseInserts(writeIterators, readIterators);
      if (zeroInit) {
        Expr sizeOfElt = Sizeof::make(queryResult.getType().getDataType());
        Expr callocValues = ir::Call::make("calloc", {size, sizeOfElt},
                                           queryResult.getType().getDataType());
        Stmt allocResult = VarDecl::make(values, callocValues);
        allocStmts.push_back(allocResult);
      }
      else {
        Stmt declResult = VarDecl::make(values, 0);
        allocStmts.push_back(declResult);

        Stmt allocResult = Allocate::make(values, size);
        allocStmts.push_back(allocResult);
      }

      Stmt freeResult = Free::make(values);
      freeStmts.push_back(freeResult);
    }
    Stmt allocResults = Block::make(allocStmts);
    freeQueryResults = Block::make(freeStmts);

    queries = lower(assemble.getQueries());
    queries = Block::blanks(allocResults, queries);
  }

  const auto& queryResults = assemble.getAttrQueryResults();
  const auto resultAccesses = getResultAccesses(assemble.getCompute()).first;

  std::vector<Stmt> initAssembleStmts;
  for (const auto& resultAccess : resultAccesses) {
    Expr prevSize = 1;
    std::vector<Expr> coords;
    const auto resultIterators = getIterators(resultAccess);
    const auto resultTensor = resultAccess.getTensorVar();
    const auto resultTensorVar = getTensorVar(resultTensor);
    const auto resultModeOrdering = resultTensor.getFormat().getModeOrdering();
    for (const auto& resultIterator : resultIterators) {
      if (generateAssembleCode()) {
        const size_t resultLevel = resultIterator.getMode().getLevel() - 1;
        const auto queryResultVars = queryResults.at(resultTensor)[resultLevel];
        std::vector<AttrQueryResult> queryResults;
        for (const auto& queryResultVar : queryResultVars) {
          queryResults.emplace_back(getTensorVar(queryResultVar),
                                    getValuesArray(queryResultVar));
        }

        if (resultIterator.hasSeqInsertEdge()) {
          Stmt initEdges = resultIterator.getSeqInitEdges(prevSize,
                                                          queryResults);
          initAssembleStmts.push_back(initEdges);

          Stmt insertEdgeLoop = resultIterator.getSeqInsertEdge(
              resultIterator.getParent().getPosVar(), coords, queryResults);
          auto locateCoords = coords;
          for (auto iter = resultIterator.getParent(); !iter.isRoot();
               iter = iter.getParent()) {
            if (iter.hasLocate()) {
              Expr dim = GetProperty::make(resultTensorVar,
                  TensorProperty::Dimension,
                  resultModeOrdering[iter.getMode().getLevel() - 1]);
              Expr pos = iter.getPosVar();
              Stmt initPos = VarDecl::make(pos, iter.locate(locateCoords)[0]);
              insertEdgeLoop = For::make(coords.back(), 0, dim, 1,
                                         Block::make(initPos, insertEdgeLoop));
            } else {
              taco_not_supported_yet;
            }
            locateCoords.pop_back();
          }
          initAssembleStmts.push_back(insertEdgeLoop);
        }

        Stmt initCoords = resultIterator.getInitCoords(prevSize, queryResults);
        initAssembleStmts.push_back(initCoords);
      }

      Stmt initYieldPos = resultIterator.getInitYieldPos(prevSize);
      initAssembleStmts.push_back(initYieldPos);

      prevSize = resultIterator.getAssembledSize(prevSize);
      coords.push_back(getCoordinateVar(resultIterator));
    }

    if (generateAssembleCode()) {
      // TODO: call calloc if not compact or not unpadded
      Expr valuesArr = getValuesArray(resultTensor);
      Stmt initValues = Allocate::make(valuesArr, prevSize);
      initAssembleStmts.push_back(initValues);
    }
  }
  Stmt initAssemble = Block::make(initAssembleStmts);

  guardedTemps = util::toSet(getTemporaries(assemble.getCompute()));
  Stmt compute = lower(assemble.getCompute());

  std::vector<Stmt> finalizeAssembleStmts;
  for (const auto& resultAccess : resultAccesses) {
    Expr prevSize = 1;
    const auto resultIterators = getIterators(resultAccess);
    for (const auto& resultIterator : resultIterators) {
      Stmt finalizeYieldPos = resultIterator.getFinalizeYieldPos(prevSize);
      finalizeAssembleStmts.push_back(finalizeYieldPos);

      prevSize = resultIterator.getAssembledSize(prevSize);
    }
  }
  Stmt finalizeAssemble = Block::make(finalizeAssembleStmts);

  return Block::blanks(queries,
                       initAssemble,
                       compute,
                       finalizeAssemble,
                       freeQueryResults);
}


Stmt LowererImpl::lowerMulti(Multi multi) {
  Stmt stmt1 = lower(multi.getStmt1());
  Stmt stmt2 = lower(multi.getStmt2());
  return Block::make(stmt1, stmt2);
}

Stmt LowererImpl::lowerSuchThat(SuchThat suchThat) {
  Stmt stmt = lower(suchThat.getStmt());
  return Block::make(stmt);
}


Expr LowererImpl::lowerAccess(Access access) {
  if (access.isAccessingStructure()) {
    return true;
  }

  TensorVar var = access.getTensorVar();

  if (isScalar(var.getType())) {
    return getTensorVar(var);
  }

  if (!getIterators(access).back().isUnique()) {
    return getReducedValueVar(access);
  }

  if (var.getType().getDataType() == Bool &&
      getIterators(access).back().isZeroless())  {
    return true;
  }

  const auto vals = getValuesArray(var);
  if (!vals.defined()) {
    return true;
  }

  if(getIterators(access).back().getPosIterKind() == taco_positer_kind::BYTE){
    auto charVals = ir::Cast::make(vals, UInt8, true);
    auto addr = Load::make(charVals, generateValueLocExpr(access), true);
    return Load::make(ir::Cast::make(addr, access.getDataType(), true));
  }

  return Load::make(vals, generateValueLocExpr(access));
}

Expr LowererImpl::lowerIndexVar(IndexVar var) {
  taco_iassert(util::contains(indexVarToExprMap, var));
  taco_iassert(provGraph.isRecoverable(var, definedIndexVars));
  return indexVarToExprMap.at(var);
}


Expr LowererImpl::lowerLiteral(Literal literal) {
  switch (literal.getDataType().getKind()) {
    case Datatype::Bool:
      return ir::Literal::make(literal.getVal<bool>());
    case Datatype::UInt8:
      return ir::Literal::make((unsigned long long)literal.getVal<uint8_t>());
    case Datatype::UInt16:
      return ir::Literal::make((unsigned long long)literal.getVal<uint16_t>());
    case Datatype::UInt32:
      return ir::Literal::make((unsigned long long)literal.getVal<uint32_t>());
    case Datatype::UInt64:
      return ir::Literal::make((unsigned long long)literal.getVal<uint64_t>());
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      return ir::Literal::make((int)literal.getVal<int8_t>());
    case Datatype::Int16:
      return ir::Literal::make((int)literal.getVal<int16_t>());
    case Datatype::Int32:
      return ir::Literal::make((int)literal.getVal<int32_t>());
    case Datatype::Int64:
      return ir::Literal::make((long long)literal.getVal<int64_t>());
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      return ir::Literal::make(literal.getVal<float>());
    case Datatype::Float64:
      return ir::Literal::make(literal.getVal<double>());
    case Datatype::Complex64:
      return ir::Literal::make(literal.getVal<std::complex<float>>());
    case Datatype::Complex128:
      return ir::Literal::make(literal.getVal<std::complex<double>>());
    case Datatype::Undefined:
      taco_unreachable;
      break;
  }
  return ir::Expr();
}


Expr LowererImpl::lowerNeg(Neg neg) {
  return ir::Neg::make(lower(neg.getA()));
}


Expr LowererImpl::lowerAdd(Add add) {
  Expr a = lower(add.getA());
  Expr b = lower(add.getB());
  return (add.getDataType().getKind() == Datatype::Bool)
         ? ir::Or::make(a, b) : ir::Add::make(a, b);
}


Expr LowererImpl::lowerSub(Sub sub) {
  return ir::Sub::make(lower(sub.getA()), lower(sub.getB()));
}


Expr LowererImpl::lowerMul(Mul mul) {
  const IndexExpr t = mul.getA();
  Expr a = lower(mul.getA());
  Expr b = lower(mul.getB());
  return (mul.getDataType().getKind() == Datatype::Bool)
         ? ir::And::make(a, b) : ir::Mul::make(a, b);
}


Expr LowererImpl::lowerDiv(Div div) {
  return ir::Div::make(lower(div.getA()), lower(div.getB()));
}


Expr LowererImpl::lowerSqrt(Sqrt sqrt) {
  return ir::Sqrt::make(lower(sqrt.getA()));
}


Expr LowererImpl::lowerCast(Cast cast) {
  return ir::Cast::make(lower(cast.getA()), cast.getDataType());
}


Expr LowererImpl::lowerCallIntrinsic(CallIntrinsic call) {
  std::vector<Expr> args;
  for (auto& arg : call.getArgs()) {
    args.push_back(lower(arg));
  }
  return call.getFunc().lower(args);
}


Expr LowererImpl::lowerTensorOp(Call op) {
  auto definedArgs = op.getDefinedArgs();
  std::vector<Expr> args;

  if(util::contains(op.getDefs(), definedArgs)) {
    auto lowerFunc = op.getDefs().at(definedArgs);
    for (auto& argIdx : definedArgs) {
      args.push_back(lower(op.getArgs()[argIdx]));
    }
    return lowerFunc(args);
  }

  for(const auto& arg : op.getArgs()) {
    args.push_back(lower(arg));
  }

  return op.getFunc()(args);
}

Stmt LowererImpl::lower(IndexStmt stmt) {
  return visitor->lower(stmt);
}


Expr LowererImpl::lower(IndexExpr expr) {
  return visitor->lower(expr);
}


bool LowererImpl::generateAssembleCode() const {
  return this->assemble;
}


bool LowererImpl::generateComputeCode() const {
  return this->compute;
}

Stmt LowererImpl::emitEarlyExit(Expr reductionExpr, std::vector<Property>& properties) {
  if (loopOrderAllowsShortCircuit && findProperty<Annihilator>(properties).defined()) {
    Literal annh = findProperty<Annihilator>(properties).annihilator();
    Expr isAnnihilator = ir::Eq::make(reductionExpr, lower(annh));
    return IfThenElse::make(isAnnihilator, Block::make(Break::make()));
  }
  return Stmt();
}

Expr LowererImpl::getTensorVar(TensorVar tensorVar) const {
  taco_iassert(util::contains(this->tensorVars, tensorVar)) << tensorVar;
  return this->tensorVars.at(tensorVar);
}


Expr LowererImpl::getCapacityVar(Expr tensor) const {
  taco_iassert(util::contains(this->capacityVars, tensor)) << tensor;
  return this->capacityVars.at(tensor);
}

vector<Expr> LowererImpl::getFillRegionVars(Expr tensor) const {
  taco_iassert(util::contains(this->fillRegionVars, tensor)) << tensor;
  return this->fillRegionVars.at(tensor);
}

ir::Expr LowererImpl::getValuesArray(TensorVar var) const
{
  return (util::contains(temporaryArrays, var))
         ? temporaryArrays.at(var).values
         : GetProperty::make(getTensorVar(var), TensorProperty::Values);
}


Expr LowererImpl::getDimension(IndexVar indexVar) const {
  taco_iassert(util::contains(this->dimensions, indexVar)) << indexVar;
  return this->dimensions.at(indexVar);
}


std::vector<Iterator> LowererImpl::getIterators(Access access) const {
  vector<Iterator> result;
  TensorVar tensor = access.getTensorVar();
  for (int i = 0; i < tensor.getOrder(); i++) {
    int mode = tensor.getFormat().getModeOrdering()[i];
    result.push_back(iterators.levelIterator(ModeAccess(access, mode+1)));
  }
  return result;
}


set<Access> LowererImpl::getExhaustedAccesses(MergePoint point,
                                              MergeLattice lattice) const
{
  set<Access> exhaustedAccesses;
  for (auto& iterator : lattice.exhausted(point)) {
    exhaustedAccesses.insert(iterators.modeAccess(iterator).getAccess());
  }
  return exhaustedAccesses;
}


Expr LowererImpl::getReducedValueVar(Access access) const {
  return this->reducedValueVars.at(access);
}


Expr LowererImpl::getCoordinateVar(IndexVar indexVar) const {
  return this->iterators.modeIterator(indexVar).getCoordVar();
}

Expr LowererImpl::getPositionVar(IndexVar indexVar) const {
  return this->iterators.modeIterator(indexVar).getPosVar();
}

Expr LowererImpl::getCoordinateVar(Iterator iterator) const {
  if (iterator.isDimensionIterator()) {
    return iterator.getCoordVar();
  }
  return this->getCoordinateVar(iterator.getIndexVar());
}


vector<Expr> LowererImpl::coordinates(Iterator iterator) const {
  taco_iassert(iterator.defined());

  vector<Expr> coords;
  do {
    coords.push_back(getCoordinateVar(iterator));
    iterator = iterator.getParent();
  } while (!iterator.isRoot());
  auto reverse = util::reverse(coords);
  return vector<Expr>(reverse.begin(), reverse.end());
}

vector<Expr> LowererImpl::parentCoordinates(Iterator iterator) const {
  auto result = coordinates(iterator);
  result.pop_back(); // We do not want to include the current coordinate
  return result;
}


vector<Expr> LowererImpl::coordinates(vector<Iterator> iterators)
{
  taco_iassert(all(iterators, [](Iterator iter){ return iter.defined(); }));
  vector<Expr> result;
  for (auto& iterator : iterators) {
    result.push_back(iterator.getCoordVar());
  }
  return result;
}

vector<Expr> LowererImpl::positions(Iterator iterator) const {
  taco_iassert(iterator.defined());

  vector<Expr> positions;
//  std::cout << "positions: ";
  do {
//    std::cout << iterator << "(" << iterator.getIteratorVar() << " " << iterator.isDimensionIterator() << ")" << ", ";
    if (iterator.isDimensionIterator()){
      positions.push_back(getCoordinateVar(iterator));
    } else {
      positions.push_back(iterator.getPosVar());
    }
    iterator = iterator.getParent();
  } while (!iterator.isRoot());
//  std::cout << std::endl;
  auto reverse = util::reverse(positions);
  return vector<Expr>(reverse.begin(), reverse.end());
}

Stmt LowererImpl::initResultArrays(vector<Access> writes,
                                   vector<Access> reads,
                                   set<Access> reducedAccesses) {

  std::vector<Stmt> result;

  multimap<IndexVar, Iterator> readIterators;
  for (auto& read : reads) {
    for (auto& readIterator : getIterators(read)) {
      for (auto& underivedAncestor : provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
        readIterators.insert({underivedAncestor, readIterator});
      }

      vector<Expr> fillVars = getFillRegionVars(readIterator.getTensor());
      Expr lenVar = fillVars[0];
      Expr indexVar = fillVars[1];
      result.push_back(VarDecl::make(lenVar, 1));
      result.push_back(VarDecl::make(indexVar, 0));
    }
  }

  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0 ||
        isAssembledByUngroupedInsertion(write.getTensorVar())) {
      continue;
    }

    std::vector<Stmt> initArrays;

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());

    Expr tensor = getTensorVar(write.getTensorVar());
    Expr fill = GetProperty::make(tensor, TensorProperty::FillValue);
    Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
    bool clearValuesAllocation = false;

    Expr parentSize = 1;
    if (generateAssembleCode()) {
      for (const auto& iterator : iterators) {
        Expr size;
        Stmt init;
        // Initialize data structures for storing levels
        if (iterator.hasAppend()) {
	        // TODO (stephenchouca): See if this makes sense.
	        if (util::getFromEnv("TACO_VALUE_ALLOC_HACK", "0") != "0") {
            size = iterator.getWidth();
	        } else {
	          size = 0;
	        }
          init = iterator.getAppendInitLevel(parentSize, size);
        } else if (iterator.hasInsert()) {
          size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
          init = iterator.getInsertInitLevel(parentSize, size);
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        initArrays.push_back(init);

        // Declare position variable of append modes that are not above a
        // branchless mode (if mode below is branchless, then can share same
        // position variable)
        if (iterator.hasAppend() && (iterator.isLeaf() ||
            !iterator.getChild().isBranchless())) {
          initArrays.push_back(VarDecl::make(iterator.getPosVar(), 0));
        }

        parentSize = size;
        // Writes into a windowed iterator require the allocation to be cleared.
        clearValuesAllocation |= (iterator.isWindowed() || iterator.hasIndexSet());
      }

      // Pre-allocate memory for the value array if computing while assembling
      if (generateComputeCode())
      {
        taco_iassert(!iterators.empty());

        Expr capacityVar = getCapacityVar(tensor);
        Expr allocSize = isValue(parentSize, 0)
                         ? DEFAULT_ALLOC_SIZE : parentSize;
        initArrays.push_back(VarDecl::make(capacityVar, allocSize));
        initArrays.push_back(Allocate::make(valuesArr, capacityVar, false /* is_realloc */, Expr() /* old_elements */,
                                            clearValuesAllocation));
      } else if (generateAssembleCode()){
        taco_iassert(!iterators.empty());

        Expr capacityVar = getCapacityVar(tensor);
        Expr allocSize = isValue(parentSize, 0)
                         ? DEFAULT_ALLOC_SIZE : parentSize;
        initArrays.push_back(VarDecl::make(capacityVar, allocSize));
      }

      taco_iassert(!initArrays.empty());
      result.push_back(Block::make(initArrays));
    }
    else if (generateComputeCode()) {
      Iterator lastAppendIterator;
      // Compute size of values array
      for (auto& iterator : iterators) {
        if (iterator.hasAppend()) {
          lastAppendIterator = iterator;
          parentSize = iterator.getSize(parentSize);
        } else if (iterator.hasInsert()) {
          parentSize = ir::Mul::make(parentSize, iterator.getWidth());
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        parentSize = simplify(parentSize);
      }

      // Declare position variable for the last append level
      if (lastAppendIterator.defined()) {
        result.push_back(VarDecl::make(lastAppendIterator.getPosVar(), 0));
      }
    }

    if (generateComputeCode() && iterators.back().hasInsert() &&
        !isValue(parentSize, 0) &&
        (hasSparseInserts(iterators, readIterators) ||
         util::contains(reducedAccesses, write))) {
      // Zero-initialize values array if size statically known and might not
      // assign to every element in values array during compute
      // TODO: Right now for scheduled code we check if any iterator is not full and then emit
      // a zero-initialization loop. We only actually need a zero-initialization loop if the combined
      // iteration of all the iterators is not full. We can check this by seeing if we can recover a
      // full iterator from our set of iterators.
      Expr size = generateAssembleCode() ? getCapacityVar(tensor) : parentSize;
      result.push_back(initValues(tensor, fill, 0, size));
    }

    for (const auto& iterator : iterators) {
      vector<Expr> fillVars = getFillRegionVars(iterator.getTensor());
      Expr lenVar = fillVars[0];
      Expr indexVar = fillVars[1];
      result.push_back(VarDecl::make(lenVar, 1));
      result.push_back(VarDecl::make(indexVar, 0));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}


ir::Stmt LowererImpl::finalizeResultArrays(std::vector<Access> writes) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  bool clearValuesAllocation = false;
  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0 ||
        isAssembledByUngroupedInsertion(write.getTensorVar())) {
      continue;
    }

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());

    Expr parentSize = 1;
    for (const auto& iterator : iterators) {
      Expr size;
      Stmt finalize;
      // Post-process data structures for storing levels
      if (iterator.hasAppend()) {
        size = iterator.getPosVar();
        Expr values = getValuesArrayFromIterator(iterator);
        finalize = iterator.getAppendFinalizeLevel(parentSize, size, values);
      } else if (iterator.hasInsert()) {
        size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
        finalize = iterator.getInsertFinalizeLevel(parentSize, size);
      } else {
        taco_ierror << "Write iterator supports neither append nor insert";
      }
      result.push_back(finalize);
      parentSize = size;
      // Writes into a windowed iterator require the allocation to be cleared.
      clearValuesAllocation |= (iterator.isWindowed() || iterator.hasIndexSet());
    }

    if (!generateComputeCode()) {
      // Allocate memory for values array after assembly if not also computing
      Expr tensor = getTensorVar(write.getTensorVar());
      Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
      result.push_back(Allocate::make(valuesArr, parentSize, false /* is_realloc */, Expr() /* old_elements */,
                                      clearValuesAllocation));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}

Stmt LowererImpl::defineScalarVariable(TensorVar var, bool zero) {
  Datatype type = var.getType().getDataType();
  Expr varValueIR = Var::make(var.getName() + "_val", type, false, false);
  Expr init = (zero) ? ir::Literal::zero(type)
                     : Load::make(GetProperty::make(tensorVars.at(var),
                                                    TensorProperty::Values));
  tensorVars.find(var)->second = varValueIR;
  return VarDecl::make(varValueIR, init);
}

static
vector<Iterator> getIteratorsFrom(IndexVar var,
                                  const vector<Iterator>& iterators) {
  vector<Iterator> result;
  bool found = false;
  for (Iterator iterator : iterators) {
    if (var == iterator.getIndexVar()) found = true;
    if (found) {
      result.push_back(iterator);
    }
  }
  return result;
}


Stmt LowererImpl::initResultArrays(IndexVar var, vector<Access> writes,
                                   vector<Access> reads,
                                   set<Access> reducedAccesses) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  multimap<IndexVar, Iterator> readIterators;
  for (auto& read : reads) {
    for (auto& readIterator : getIteratorsFrom(var, getIterators(read))) {
      for (auto& underivedAncestor : provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
        readIterators.insert({underivedAncestor, readIterator});
      }
    }
  }

  vector<Stmt> result;
  for (auto& write : writes) {
    Expr tensor = getTensorVar(write.getTensorVar());
    Expr fill = GetProperty::make(tensor, TensorProperty::FillValue);
    Expr values = GetProperty::make(tensor, TensorProperty::Values);

    vector<Iterator> iterators = getIteratorsFrom(var, getIterators(write));

    if (iterators.empty()) {
      continue;
    }

    Iterator resultIterator = iterators.front();

    // Initialize begin var
    if (resultIterator.hasAppend() && !resultIterator.isBranchless()) {
      Expr begin = resultIterator.getBeginVar();
      result.push_back(VarDecl::make(begin, resultIterator.getPosVar()));
    }

    const bool isTopLevel = (iterators.size() == write.getIndexVars().size());
    if (resultIterator.getParent().hasAppend() || isTopLevel) {
      Expr resultParentPos = resultIterator.getParent().getPosVar();
      Expr resultParentPosNext = simplify(ir::Add::make(resultParentPos, 1));
      Expr initBegin = resultParentPos;
      Expr initEnd = resultParentPosNext;
      Expr stride = 1;

      Iterator initIterator;
      for (Iterator iterator : iterators) {
        if (!iterator.hasInsert()) {
          initIterator = iterator;
          break;
        }

        stride = simplify(ir::Mul::make(stride, iterator.getWidth()));
        initBegin = simplify(ir::Mul::make(resultParentPos, stride));
        initEnd = simplify(ir::Mul::make(resultParentPosNext, stride));

        // Initialize data structures for storing insert mode
        result.push_back(iterator.getInsertInitCoords(initBegin, initEnd));
      }

      if (initIterator.defined()) {
        // Initialize data structures for storing edges of next append mode
        taco_iassert(initIterator.hasAppend());
        result.push_back(initIterator.getAppendInitEdges(initBegin, initEnd));
      } else if (generateComputeCode() && !isTopLevel) {
        if (isa<ir::Mul>(stride)) {
          Expr strideVar = Var::make(util::toString(tensor) + "_stride", Int());
          result.push_back(VarDecl::make(strideVar, stride));
          stride = strideVar;
        }

        // Resize values array if not large enough
        Expr capacityVar = getCapacityVar(tensor);
        Expr size = simplify(ir::Mul::make(resultParentPosNext, stride));
        result.push_back(atLeastDoubleSizeIfFull(values, capacityVar, size));

        if (hasSparseInserts(iterators, readIterators) ||
            util::contains(reducedAccesses, write)) {
          // Zero-initialize values array if might not assign to every element
          // in values array during compute
          result.push_back(initValues(tensor, fill, resultParentPos, stride));
        }
      }
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::resizeAndInitValues(const std::vector<Iterator>& appenders,
                                      const std::set<Access>& reducedAccesses) {
  if (!generateComputeCode()) {
    return Stmt();
  }

  std::function<Expr(Access)> getTensor = [&](Access access) {
    return getTensorVar(access.getTensorVar());
  };
  const auto reducedTensors = util::map(reducedAccesses, getTensor);

  std::vector<Stmt> result;

  for (auto& appender : appenders) {
    if (!appender.isLeaf()) {
      continue;
    }

    Expr tensor = appender.getTensor();
    Expr values = GetProperty::make(tensor, TensorProperty::Values);
    Expr capacity = getCapacityVar(appender.getTensor());
    Expr pos = appender.getPosVar();

    if (generateAssembleCode()) {
      result.push_back(doubleSizeIfFull(values, capacity, pos));
    }

    if (util::contains(reducedTensors, tensor)) {
      Expr zero = ir::Literal::zero(tensor.type());
      result.push_back(Store::make(values, pos, zero));
    }
  }

  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::initValues(Expr tensor, Expr initVal, Expr begin, Expr size) {
  Expr lower = simplify(ir::Mul::make(begin, size));
  Expr upper = simplify(ir::Mul::make(ir::Add::make(begin, 1), size));
  Expr p = Var::make("p" + util::toString(tensor), Int());
  Expr values = GetProperty::make(tensor, TensorProperty::Values);
  Stmt zeroInit = Store::make(values, p, initVal);
  LoopKind parallel = (isa<ir::Literal>(size) && 
                       to<ir::Literal>(size)->getIntValue() < (1 << 10))
                      ? LoopKind::Serial : LoopKind::Static_Chunked;
  if (should_use_CUDA_codegen() && util::contains(parallelUnitSizes, ParallelUnit::GPUBlock)) {
    return ir::VarDecl::make(ir::Var::make("status", Int()),
                         ir::Call::make("cudaMemset", {values, ir::Literal::make(0, Int()),
                             ir::Mul::make(ir::Sub::make(upper, lower),
                             ir::Literal::make(values.type().getNumBytes()))}, Int()));
  }
  return For::make(p, lower, upper, 1, zeroInit, parallel);
}

Stmt LowererImpl::declLocatePosVars(vector<Iterator> locators) {
  vector<Stmt> result;
  for (Iterator& locator : locators) {
    accessibleIterators.insert(locator);

    bool doLocate = true;
    for (Iterator ancestorIterator = locator.getParent();
         !ancestorIterator.isRoot() && ancestorIterator.hasLocate();
         ancestorIterator = ancestorIterator.getParent()) {
      if (!accessibleIterators.contains(ancestorIterator)) {
        doLocate = false;
      }
    }

    if (doLocate) {
      Iterator locateIterator = locator;
      if (locateIterator.hasPosIter()) {
        taco_iassert(!provGraph.isUnderived(locateIterator.getIndexVar()));
        continue; // these will be recovered with separate procedure
      }
      do {
        auto coords = coordinates(locateIterator);
        // If this dimension iterator operates over a window, then it needs
        // to be projected up to the window's iteration space.
        if (locateIterator.isWindowed()) {
          auto expr = coords[coords.size() - 1];
          coords[coords.size() - 1] = this->projectCanonicalSpaceToWindowedPosition(locateIterator, expr);
        } else if (locateIterator.hasIndexSet()) {
          // If this dimension iterator operates over an index set, follow the
          // indirection by using the locator access the index set's crd array.
          // The resulting value is where we should locate into the actual tensor.
          auto expr = coords[coords.size() - 1];
          auto indexSetIterator = locateIterator.getIndexSetIterator();
          Expr values = getValuesArrayFromIterator(locateIterator);
          //TODO: Check what the positions vector should be
          auto coordArray = indexSetIterator.posAccess({expr}, coordinates(indexSetIterator), values,
                                                       tensorTypes[locateIterator.getTensor()].getDataType()).getResults()[0];
          coords[coords.size() - 1] = coordArray;
        }
        ModeFunction locate = locateIterator.locate(coords);
        taco_iassert(isValue(locate.getResults()[1], true));
        Stmt declarePosVar = VarDecl::make(locateIterator.getPosVar(),
                                           locate.getResults()[0]);
        result.push_back(declarePosVar);

        if (locateIterator.isLeaf()) {
          break;
        }
        locateIterator = locateIterator.getChild();
      } while (accessibleIterators.contains(locateIterator));
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::reduceDuplicateCoordinates(Expr coordinate,
                                             vector<Iterator> iterators,
                                             bool alwaysReduce) {
  vector<Stmt> result;
  for (Iterator& iterator : iterators) {
    taco_iassert(!iterator.isUnique() && iterator.hasPosIter());

    Access access = this->iterators.modeAccess(iterator).getAccess();
    Expr iterVar = iterator.getIteratorVar();
    Expr segendVar = iterator.getSegendVar();
    Expr reducedVal = iterator.isLeaf() ? getReducedValueVar(access) : Expr();
    Expr tensorVar = getTensorVar(access.getTensorVar());
    Expr tensorVals = GetProperty::make(tensorVar, TensorProperty::Values);

    // Initialize variable storing reduced component value.
    if (reducedVal.defined()) {
      Expr reducedValInit = alwaysReduce
                          ? Load::make(tensorVals, iterVar)
                          : ir::Literal::zero(reducedVal.type());
      result.push_back(VarDecl::make(reducedVal, reducedValInit));
    }

    if (iterator.isLeaf()) {
      // If iterator is over bottommost coordinate hierarchy level and will
      // always advance (i.e., not merging with another iterator), then we don't
      // need a separate segend variable.
      segendVar = iterVar;
      if (alwaysReduce) {
        result.push_back(addAssign(segendVar, 1));
      }
    } else {
      Expr segendInit = alwaysReduce ? ir::Add::make(iterVar, 1) : iterVar;
      result.push_back(VarDecl::make(segendVar, segendInit));
    }

    vector<Stmt> dedupStmts;
    if (reducedVal.defined()) {
      Expr partialVal = Load::make(tensorVals, segendVar);
      dedupStmts.push_back(addAssign(reducedVal, partialVal));
    }
    dedupStmts.push_back(addAssign(segendVar, 1));
    Stmt dedupBody = Block::make(dedupStmts);

    // TODO: Cast to byte-array values
    ModeFunction posAccess = iterator.posAccess({segendVar},
                                                coordinates(iterator),
                                                tensorVals, tensorTypes[iterator.getTensor()].getDataType());
    // TODO: Support access functions that perform additional computations
    //       and/or might access invalid positions.
    taco_iassert(!posAccess.compute().defined());
    taco_iassert(to<ir::Literal>(posAccess.getResults()[1])->getBoolValue());
    Expr nextCoord = posAccess.getResults()[0];
    Expr withinBounds = Lt::make(segendVar, iterator.getEndVar());
    Expr isDuplicate = Eq::make(posAccess.getResults()[0], coordinate);
    result.push_back(While::make(And::make(withinBounds, isDuplicate),
                                 Block::make(dedupStmts)));
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImpl::codeToInitializeIteratorVar(Iterator iterator, vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> result;
  taco_iassert(iterator.hasPosIter() || iterator.hasCoordIter() ||
               iterator.isDimensionIterator());

  Expr iterVar = iterator.getIteratorVar();
  Expr endVar = iterator.getEndVar();
  if (iterator.hasPosIter()) {
    Expr parentPos = iterator.getParent().getPosVar();
    if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
      // E.g. a compressed mode without duplicates
      ModeFunction bounds = iterator.posBounds(parentPos);
      result.push_back(bounds.compute());
      // if has a coordinate ranger then need to binary search
      if (any(rangers,
              [](Iterator it){ return it.isDimensionIterator(); })) {

        Expr binarySearchTarget = provGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[coordinateVar][0];
        if (binarySearchTarget != underivedBounds[coordinateVar][0]) {
          // If we have a window, then we need to project up the binary search target
          // into the window rather than the beginning of the level.
          if (iterator.isWindowed()) {
            binarySearchTarget = this->projectCanonicalSpaceToWindowedPosition(iterator, binarySearchTarget);
          }
          result.push_back(VarDecl::make(iterator.getBeginVar(), binarySearchTarget));

          vector<Expr> binarySearchArgs = {
                  iterator.getMode().getModePack().getArray(1), // array
                  bounds[0], // arrayStart
                  bounds[1], // arrayEnd
                  iterator.getBeginVar() // target
          };
          result.push_back(
                  VarDecl::make(iterVar, ir::Call::make("taco_binarySearchAfter", binarySearchArgs, iterVar.type())));
        }
        else {
          result.push_back(VarDecl::make(iterVar, bounds[0]));
        }
      }
      else {
        auto bound = bounds[0];
        // If we have a window on this iterator, then search for the start of
        // the window rather than starting at the beginning of the level.
        if (iterator.isWindowed()) {
            bound = this->searchForStartOfWindowPosition(iterator, bounds[0], bounds[1]);
        }
        result.push_back(VarDecl::make(iterVar, bound));
      }

      result.push_back(VarDecl::make(endVar, bounds[1]));
    } else {
      taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
      taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

      // E.g. a compressed mode with duplicates. Apply iterator chaining
      Expr parentSegend = iterator.getParent().getSegendVar();
      ModeFunction startBounds = iterator.posBounds(parentPos);
      ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
      result.push_back(startBounds.compute());
      result.push_back(VarDecl::make(iterVar, startBounds[0]));
      result.push_back(endBounds.compute());
      result.push_back(VarDecl::make(endVar, endBounds[1]));
    }
  }
  else if (iterator.hasCoordIter()) {
    // E.g. a hasmap mode
    vector<Expr> coords = coordinates(iterator);
    coords.erase(coords.begin());
    ModeFunction bounds = iterator.coordBounds(coords);
    result.push_back(VarDecl::make(coords.back(), 0)); //TODO: This is a hack to get things to work
    result.push_back(bounds.compute());
    result.push_back(VarDecl::make(iterVar, bounds[0]));
    result.push_back(VarDecl::make(endVar, bounds[1]));
  }
  else if (iterator.isDimensionIterator()) {
    // A dimension
    // If a merger then initialize to 0
    // If not then get first coord value like doing normal merge

    // If derived then need to recoverchild from this coord value
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr coord = coordinates(vector<Iterator>({iterator}))[0];
      result.push_back(VarDecl::make(coord, 0));
    }
    else {
      result.push_back(codeToLoadCoordinatesFromPosIterators(iterators, true, coordinate));

      Stmt stmt = resolveCoordinate(mergers, coordinate, true);
      taco_iassert(stmt != Stmt());
      result.push_back(stmt);
      result.push_back(codeToRecoverDerivedIndexVar(coordinateVar, iterator.getIndexVar(), true));

      // emit bound for ranger too
      vector<Expr> startBounds;
      vector<Expr> endBounds;
      for (Iterator merger : mergers) {
        ModeFunction coordBounds = merger.coordBounds(merger.getParent().getPosVar());
        startBounds.push_back(coordBounds[0]);
        endBounds.push_back(coordBounds[1]);
      }
      //TODO: maybe needed after split reorder? underivedBounds[coordinateVar] = {ir::Max::make(startBounds), ir::Min::make(endBounds)};
      Stmt end_decl = VarDecl::make(iterator.getEndVar(), provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[1]);
      result.push_back(end_decl);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImpl::codeToInitializeIteratorVars(vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> results;
  // initialize mergers first (can't depend on initializing rangers)
  for (Iterator iterator : mergers) {
    results.push_back(codeToInitializeIteratorVar(iterator, iterators, rangers, mergers, coordinate, coordinateVar));
  }

  for (Iterator iterator : rangers) {
      if (find(mergers.begin(), mergers.end(), iterator) == mergers.end()) {
        results.push_back(codeToInitializeIteratorVar(iterator, iterators, rangers, mergers, coordinate, coordinateVar));
      }
  }
  return results.empty() ? Stmt() : Block::make(results);
}

Stmt LowererImpl::codeToRecoverDerivedIndexVar(IndexVar underived, IndexVar indexVar, bool emitVarDecl) {
  if(underived != indexVar) {
    // iterator indexVar must be derived from coordinateVar
    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(indexVar);
    taco_iassert(find(underivedAncestors.begin(), underivedAncestors.end(), underived) != underivedAncestors.end());

    vector<Stmt> recoverySteps;
    for (const IndexVar& varToRecover : provGraph.derivationPath(underived, indexVar)) {
      if(varToRecover == underived) continue;
      recoverySteps.push_back(provGraph.recoverChild(varToRecover, indexVarToExprMap, emitVarDecl, iterators));
    }
    return Block::make(recoverySteps);
  }
  return Stmt();
}

Stmt LowererImpl::codeToIncIteratorVars(Expr coordinate, IndexVar coordinateVar, vector<Iterator> iterators, vector<Iterator> mergers) {
  if (iterators.size() == 1) {
    Expr ivar = iterators[0].getIteratorVar();

    Datatype type = tensorTypes[iterators[0].getTensor()].getDataType();
    int increment = iterators[0].getPosIterKind() == taco_positer_kind::BYTE ? type.getNumBytes() : 1;

    if (iterators[0].isUnique()) {
      return addAssign(ivar, increment);
    }

    // If iterator is over bottommost coordinate hierarchy level with
    // duplicates and iterator will always advance (i.e., not merging with
    // another iterator), then deduplication loop will take care of
    // incrementing iterator variable.
    return iterators[0].isLeaf()
           ? Stmt()
           : Assign::make(ivar, iterators[0].getSegendVar());
  }

  vector<Stmt> result;

  // We emit the level iterators before the mode iterator because the coordinate
  // of the mode iterator is used to conditionally advance the level iterators.

  auto levelIterators =
      filter(iterators, [](Iterator it){return !it.isDimensionIterator();});
  for (auto& iterator : levelIterators) {
    Expr ivar = iterator.getIteratorVar();
    if (iterator.isUnique()) {
      Datatype type = tensorTypes[iterator.getTensor()].getDataType();
      int incrementVal = iterator.getPosIterKind() == taco_positer_kind::BYTE ? type.getNumBytes() : 1;

      if (iterator.isLastValueFill() && !iterator.isFull()){
        if(iterator.isLeaf()){
          auto assign = Assign::make(GetProperty::make(iterator.getTensor(), TensorProperty::FillValue),
                                     Load::make(GetProperty::make(iterator.getTensor(), TensorProperty::Values),
                                                ivar));

          auto ifstmt = ir::IfThenElse::make(Eq::make(iterator.getCoordVar(), coordinate),
                                             Block::make(assign, addAssign(ivar, 1)));

          result.push_back(ifstmt);
        } else {
          auto assign = Assign::make(GetProperty::make(iterator.getTensor(), TensorProperty::FillValue),
                                     Load::make(GetProperty::make(iterator.getTensor(), TensorProperty::Values),
                                                ivar));

          auto ifstmt = ir::IfThenElse::make(Eq::make(iterator.getCoordVar(), coordinate),
                                             Block::make(assign, addAssign(ivar, 1)));

          result.push_back(ifstmt);
        }
      } else {

        Expr increment = iterator.isFull()
                         ? 1
                         : ir::Cast::make(Eq::make(iterator.getCoordVar(),
                                                   coordinate),
                                          ivar.type());

        if(iterator.getTensor().defined()) {
          Expr values = getValuesArrayFromIterator(iterator);
          auto posAccess = iterator.posAccess(positions(iterator), coordinates(iterator),
                                              values, tensorTypes[iterator.getTensor()].getDataType());
//          if (posAccess[1].as<ir::Literal>() == nullptr) {
//            increment =  ir::Cast::make(ir::And::make(Eq::make(iterator.getCoordVar(), coordinate),
//                                                      posAccess[1]), ivar.type()); // Need to check that the coordinate has been found also
//          }
        }


        result.push_back(addAssign(ivar, ir::Mul::make(increment, incrementVal)));
      }
      if (isMultiPosIter(iterator)){
        Expr increment = iterator.isFull()
                         ? 1
                         : ir::Cast::make(Eq::make(iterator.getCoordVar(),
                                                   coordinate),
                                          ivar.type());
        result.push_back(addAssign(iterator.getCoordVar(), increment));
      }
    } else if (!iterator.isLeaf()) {
      result.push_back(Assign::make(ivar, iterator.getSegendVar()));
    }
  }

  auto modeIterators =
      filter(iterators, [](Iterator it){return it.isDimensionIterator();});
  for (auto& iterator : modeIterators) {
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr ivar = iterator.getIteratorVar();
      result.push_back(addAssign(ivar, 1));
    }
    else {
      result.push_back(codeToLoadCoordinatesFromPosIterators(iterators, false, coordinate));
      Stmt stmt = resolveCoordinate(mergers, coordinate, false);
      taco_iassert(stmt != Stmt());
      result.push_back(stmt);
      result.push_back(codeToRecoverDerivedIndexVar(coordinateVar, iterator.getIndexVar(), false));
    }
  }

  return Block::make(result);
}

class ForallVisitor : public IndexNotationVisitor {

public:
    ForallVisitor() {}

    bool check(const IndexStmt& expr) {
      expr.accept(this);
      return anyForalls;
    }

private:
    bool anyForalls = false;
    std::vector<TensorVar> tensorFills;

    using IndexNotationVisitor::visit;

    void visit(const ForallNode* node){
      anyForalls = true;
      IndexNotationVisitor::visit(node);
    }
};

bool LowererImpl::checkUpdateFillRequired(Iterator& it){
  Expr values = getValuesArrayFromIterator(it);
  ModeFunction posAccess = it.posAccess(positions(it),
                                        coordinates(it),
                                        values, tensorTypes[it.getTensor()].getDataType());
  ModeFunction fillUpdate = it.getFillRegion(it.getPosVar(),
                                             coordinates(it),
                                             values, tensorTypes[it.getTensor()].getDataType());
  if (posAccess.defined() &&
      posAccess.getResults().size() > 2 &&
      posAccess[3].as<ir::Literal>() &&
      posAccess[3].as<ir::Literal>()->equalsScalar(1)){
    return false;
  }
  if (it.updatesFillRegion() &&
      fillUpdate.defined() &&
      fillUpdate.getResults().size() == 3 &&
      fillUpdate[1].as<ir::Literal>() &&
      fillUpdate[1].as<ir::Literal>()->equalsScalar(1)){
    return false;
  }
  return true;
}

Stmt LowererImpl::codeToUpdateFills(IndexStmt statement, Expr coordinate, IndexVar coordinateVar,
                                    vector<Iterator> iterators, vector<Iterator> mergers) {
  vector<Stmt> result;
  if (!ForallVisitor().check(statement)) {
    auto res = AllFillsVisitor().check(statement);
    for (size_t i=0; i < std::get<1>(res).size(); i++){
      auto& tensorVar = std::get<1>(res)[i];
      auto& access = std::get<2>(res)[i];
      auto accessIters = getIterators(access);
      bool requireUpdate = false;
      for (auto& it : accessIters){
        if (it.updatesFillRegion()){
          requireUpdate = true;
          if (!checkUpdateFillRequired(it)) requireUpdate = false;
        }
      }

      if (requireUpdate){
        auto fillRegion = GetProperty::make(getTensorVar(tensorVar), TensorProperty::FillRegion);
        auto fillVars = getFillRegionVars(getTensorVar(tensorVar));
        auto lengthVar = fillVars[0];
        auto indexVar = fillVars[1];

        // We must also update the fill variable from the fill region
        auto fillVariable = GetProperty::make(getTensorVar(tensorVar), TensorProperty::FillValue);
//        result.push_back(Assign::make(fillVariable, Load::make(fillRegion, indexVar)));
//        result.push_back(Assign::make(indexVar, ir::Rem::make(ir::Add::make(indexVar, 1), lengthVar)));

        auto ifLengthOne = IfThenElse::make(ir::Gt::make(lengthVar,1),
                                            Block::make(Assign::make(fillVariable, Load::make(fillRegion, indexVar)),
                                                        Assign::make(indexVar, ir::Add::make(indexVar, 1)),
                                                        IfThenElse::make(ir::Eq::make(indexVar, lengthVar), Assign::make(indexVar, 0))
                                            ));
        result.push_back(ifLengthOne);
      }
    }
  };
  return Block::make(result);
}

Stmt LowererImpl::codeToLoadCoordinatesFromPosIterators(vector<Iterator> iterators, bool declVars,
                                                        ir::Expr coordinate,
                                                        bool checkCount, bool afterResolve) {
  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates;
  if (iterators.size() > 1) {
    vector<Stmt> loadPosIterCoordinateStmts;
    auto posIters = filter(iterators, [](Iterator it){return it.hasPosIter();});
    for (auto& posIter : posIters) {
      vector<Stmt> stmts;
      taco_tassert(posIter.hasPosIter());
      Expr values = getValuesArrayFromIterator(posIter);
      auto fillRegion = GetProperty::make(posIter.getTensor(), TensorProperty::FillRegion);
      auto fillVariable = GetProperty::make(posIter.getTensor(), TensorProperty::FillValue);

      ModeFunction posAccess = posIter.posAccess(positions(posIter),
                                                 coordinates(posIter),
                                                 values, tensorTypes[posIter.getTensor()].getDataType());
      stmts.push_back(posAccess.compute());
      auto access = posAccess[0];
      auto coordFound = posAccess[1];
      if (posAccess.getResults().size() > 2){

        auto fillRegionReplacement = posAccess[2];
        auto length = posAccess[3];
        auto updateFill = posAccess[4];


        auto fillVars = getFillRegionVars(posIter.getTensor());
        auto lengthVar = fillVars[0];
        auto indexVar = fillVars[1];

        if (length.as<ir::Literal>() && length.as<ir::Literal>()->equalsScalar(1)){
          auto body = Block::make(Assign::make(fillVariable, Load::make(fillRegionReplacement)));
          stmts.push_back(IfThenElse::make(updateFill, body));
        } else {
          auto body = Block::make(
                  Assign::make(lengthVar, length),
                  Assign::make(indexVar, ternaryOp(ir::Eq::make(lengthVar,1), 0, 1)),
//                  IfThenElse::make(ir::Eq::make(lengthVar,1), Assign::make(indexVar,0), Assign::make(indexVar,1)),
                  Assign::make(fillRegion, fillRegionReplacement),
                  Assign::make(fillVariable, Load::make(fillRegion, 0)));
          stmts.push_back(IfThenElse::make(updateFill, body));
        }
      }
//      if (!(ir::isa<ir::Literal>(coordFound) && ir::to<ir::Literal>(coordFound)->getBoolValue()))
//      {
//        taco_iassert(posIter.isUnique()); // This check is from the single iterator logic in codeToIncIteratorVars
//        auto coordLoop = While::make(ir::Neg::make(coordFound), Block::make(addAssign(posIter.getPosVar(), 1), posAccess.compute())); // TODO: Needs to search for coordinates
//        stmts.push_back(coordLoop);
//      }
      // If this iterator is windowed, then it needs to be projected down to
      // recover the coordinate variable.
      // TODO (rohany): Would be cleaner to have this logic be moved into the
      //  ModeFunction, rather than having to check in some places?
      if (posIter.isWindowed()) {

        // If the iterator is strided, then we have to skip over coordinates
        // that don't match the stride. To do that, we insert a guard on the
        // access. We first extract the access into a temp to avoid emitting
        // a duplicate load on the _crd array.
        if (posIter.isStrided()) {
          stmts.push_back(VarDecl::make(posIter.getWindowVar(), access));
          access = posIter.getWindowVar();
          // Since we're locating into a compressed array (not iterating over it),
          // we need to advance the outer loop if the current coordinate is not
          // along the desired stride. So, we pass true to the incrementPosVar
          // argument of strideBoundsGuard.
          stmts.push_back(this->strideBoundsGuard(posIter, access, true /* incrementPosVar */));
        }

        access = this->projectWindowedPositionToCanonicalSpace(posIter, access);
      }
      if (declVars) {
        stmts.push_back(VarDecl::make(posIter.getCoordVar(), access));
      } else {
        stmts.push_back(Assign::make(posIter.getCoordVar(), access));
      }
      if (checkCount && isMultiPosIter(posIter)) {
        stmts.push_back(Assign::make(posIter.getCountVar(), ir::Cast::make(coordFound, Int())));
      }
      if (posIter.isWindowed()) {
        stmts.push_back(this->upperBoundGuardForWindowPosition(posIter, posIter.getCoordVar()));
      }

      if(checkCount && isMultiPosIter(posIter)){
        auto coordCheck = ir::Eq::make(coordinate, posIter.getCoordVar());
        auto countCheck = ir::Neg::make(ir::Cast::make(posIter.getCountVar(), Bool));
        loadPosIterCoordinateStmts.push_back(
                IfThenElse::make(ir::And::make(coordCheck, countCheck), Block::make(stmts)));
      } else {
        loadPosIterCoordinateStmts.push_back(Block::make(stmts));
      }
    }
    loadPosIterCoordinates = Block::make(loadPosIterCoordinateStmts);
  }
  return loadPosIterCoordinates;
}

Stmt LowererImpl::codeToLoadPositionsFromCoordIterators(vector<Iterator> iterators, bool declVars) {
  Stmt loads;
  if(iterators.size() > 1) {
    vector<Stmt> blockInstrs;
    auto coordIters = filter(iterators, [](Iterator it) { return it.hasCoordIter(); });
    for (auto &coordIter : coordIters) {
      taco_tassert(coordIter.hasCoordIter());
      ModeFunction coordAccess = coordIter.coordAccess(coordinates(coordIter));
      blockInstrs.push_back(coordAccess.compute());
      // TODO: This assumes that the access function always succeeds
      if (declVars) {
        blockInstrs.push_back(VarDecl::make(coordIter.getPosVar(),
                                            coordAccess[0]));
      } else {
        blockInstrs.push_back(Assign::make(coordIter.getPosVar(),
                                           coordAccess[0]));
      }
    }
    return Block::make(blockInstrs);
  }
  return Stmt();
}

static
bool isLastAppender(Iterator iter) {
  taco_iassert(iter.hasAppend());
  while (!iter.isLeaf()) {
    iter = iter.getChild();
    if (iter.hasAppend()) {
      return false;
    }
  }
  return true;
}


Stmt LowererImpl::appendCoordinate(vector<Iterator> appenders, Expr coord) {
  vector<Stmt> result;
  for (auto& appender : appenders) {
    Expr pos = appender.getPosVar();
    Iterator appenderChild = appender.getChild();

    if (appenderChild.defined() && appenderChild.isBranchless()) {
      // Already emitted assembly code for current level when handling
      // branchless child level, so don't emit code again.
      continue;
    }

    vector<Stmt> appendStmts;

    if (generateAssembleCode()) {
      Expr values = getValuesArrayFromIterator(appender);
      appendStmts.push_back(appender.getAppendCoord(pos, coord, values, getCapacityVar(appender.getTensor()),
                                                    tensorTypes[appender.getTensor()].getDataType()));
      while (!appender.isRoot() && appender.isBranchless()) {
        // Need to append result coordinate to parent level as well if child
        // level is branchless (so child coordinates will have unique parents).
        appender = appender.getParent();
        if (!appender.isRoot()) {
          taco_iassert(appender.hasAppend()) << "Parent level of branchless, "
              << "append-capable level must also be append-capable";
          taco_iassert(!appender.isUnique()) << "Need to be able to insert "
              << "duplicate coordinates to level, but level is declared unique";

          appendStmts.push_back(appender.getAppendCoord(pos, coord, values, getCapacityVar(appender.getTensor()),
                                                        tensorTypes[appender.getTensor()].getDataType()));
        }
      }
    }

    if (generateAssembleCode() || isLastAppender(appender)) {
      Datatype type = tensorTypes[appender.getTensor()].getDataType();
      int increment = appender.getPosIterKind() == taco_positer_kind::BYTE ? type.getNumBytes() : 1;

      appendStmts.push_back(addAssign(pos, increment));

      Stmt appendCode = Block::make(appendStmts);
      if (appenderChild.defined() && appenderChild.hasAppend()) {
        // Emit guard to avoid appending empty slices to result.
        // TODO: Users should be able to configure whether to append zeroes.
        Expr shouldAppend = Lt::make(appenderChild.getBeginVar(),
                                     appenderChild.getPosVar());
        appendCode = IfThenElse::make(shouldAppend, appendCode);
      }
      result.push_back(appendCode);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::generateAppendPositions(vector<Iterator> appenders) {
  vector<Stmt> result;
  if (generateAssembleCode()) {
    for (Iterator appender : appenders) {
      if (!appender.isBranchless()) {
        Expr pos = [](Iterator appender) {
          // Get the position variable associated with the appender. If a mode
          // is above a branchless mode, then the two modes can share the same
          // position variable.
          while (!appender.isLeaf() && appender.getChild().isBranchless()) {
            appender = appender.getChild();
          }
          return appender.getPosVar();
        }(appender);
        Expr beginPos = appender.getBeginVar();
        Expr parentPos = appender.getParent().getPosVar();
        result.push_back(appender.getAppendEdges(parentPos, beginPos, pos));
      }
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Expr LowererImpl::generateValueLocExpr(Access access) const {
  if (isScalar(access.getTensorVar().getType())) {
    return ir::Literal::make(0);
  }
  Iterator it = getIterators(access).back();

  // to make indexing temporary arrays with index var work correctly
  if (!provGraph.isUnderived(it.getIndexVar()) && !access.getIndexVars().empty() &&
      util::contains(indexVarToExprMap, access.getIndexVars().front()) &&
      !it.hasPosIter() && access.getIndexVars().front() == it.getIndexVar()) {
    return indexVarToExprMap.at(access.getIndexVars().front());
  }

  return it.getPosVar();
}


Expr LowererImpl::checkThatNoneAreExhausted(std::vector<Iterator> iterators)
{
  taco_iassert(!iterators.empty());
  if (iterators.size() == 1 && iterators[0].isFull()) {
    std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(iterators[0].getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators);
    Expr guards = Lt::make(iterators[0].getIteratorVar(), bounds[1]);
    if (bounds[0] != ir::Literal::make(0)) {
      guards = And::make(guards, Gte::make(iterators[0].getIteratorVar(), bounds[0]));
    }
    return guards;
  }

  vector<Expr> result;
  for (const auto& iterator : iterators) {
    taco_iassert(!iterator.isFull()) << iterator
        << " - full iterators do not need to partake in merge loop bounds";
    Expr iterUnexhausted = Lt::make(iterator.getIteratorVar(),
                                    iterator.getEndVar());
    result.push_back(iterUnexhausted);
  }

  return (!result.empty())
         ? taco::ir::conjunction(result)
         : Lt::make(iterators[0].getIteratorVar(), iterators[0].getEndVar());
}


Expr LowererImpl::generateAssembleGuard(IndexExpr expr) {
  class GenerateGuard : public IndexExprVisitorStrict {
  public:
    GenerateGuard(const std::set<TensorVar>& guardedTemps,
                  const std::map<TensorVar,Expr>& tempToGuard)
        : guardedTemps(guardedTemps), tempToGuard(tempToGuard) {}

    Expr lower(IndexExpr expr) {
      this->expr = Expr();
      IndexExprVisitorStrict::visit(expr);
      return this->expr;
    }

  private:
    Expr expr;
    const std::set<TensorVar>& guardedTemps;
    const std::map<TensorVar,Expr>& tempToGuard;

    using IndexExprVisitorStrict::visit;

    void visit(const AccessNode* node) {
      expr = (util::contains(guardedTemps, node->tensorVar) &&
              node->tensorVar.getOrder() == 0)
             ? tempToGuard.at(node->tensorVar) : true;
    }

    void visit(const LiteralNode* node) {
      expr = true;
    }

    void visit(const NegNode* node) {
      expr = lower(node->a);
    }

    void visit(const AddNode* node) {
      expr = Or::make(lower(node->a), lower(node->b));
    }

    void visit(const SubNode* node) {
      expr = Or::make(lower(node->a), lower(node->b));
    }

    void visit(const MulNode* node) {
      expr = And::make(lower(node->a), lower(node->b));
    }

    void visit(const DivNode* node) {
      expr = And::make(lower(node->a), lower(node->b));
    }

    void visit(const SqrtNode* node) {
      expr = lower(node->a);
    }

    void visit(const CastNode* node) {
      expr = lower(node->a);
    }

    void visit(const CallIntrinsicNode* node) {
      Expr ret = false;
      for (const auto& arg : node->args) {
        ret = Or::make(ret, lower(arg));
      }
      expr = ret;
    }

    void visit(const ReductionNode* node) {
      taco_unreachable;
    }

    void visit(const CallNode* node) {
      Expr ret = false;
      for (const auto& arg : node->args) {
        ret = Or::make(ret, lower(arg));
      }
      expr = ret;
    }

    void visit(const IndexVarNode* node) {
      expr = true;
    }
  };

  return ir::simplify(GenerateGuard(guardedTemps, tempToBitGuard).lower(expr));
}


bool LowererImpl::isAssembledByUngroupedInsertion(TensorVar result) {
  return util::contains(assembledByUngroupedInsert, result);
}


bool LowererImpl::isAssembledByUngroupedInsertion(Expr result) {
  for (const auto& tensor : assembledByUngroupedInsert) {
    if (getTensorVar(tensor) == result) {
      return true;
    }
  }
  return false;
}


bool LowererImpl::hasStores(Stmt stmt) {
  if (!stmt.defined()) {
    return false;
  }

  struct FindStores : IRVisitor {
    bool hasStore;
    const std::map<TensorVar, Expr>& tensorVars;
    const std::map<TensorVar, Expr>& tempToBitGuard;

    using IRVisitor::visit;

    FindStores(const std::map<TensorVar, Expr>& tensorVars,
               const std::map<TensorVar, Expr>& tempToBitGuard)
        : tensorVars(tensorVars), tempToBitGuard(tempToBitGuard) {}

    void visit(const Store* stmt) {
      hasStore = true;
    }

    void visit(const Assign* stmt) {
      for (const auto& tensorVar : tensorVars) {
        if (stmt->lhs == tensorVar.second) {
          hasStore = true;
          break;
        }
      }
      if (hasStore) {
        return;
      }
      for (const auto& bitGuard : tempToBitGuard) {
        if (stmt->lhs == bitGuard.second) {
          hasStore = true;
          break;
        }
      }
    }

    bool hasStores(Stmt stmt) {
      hasStore = false;
      stmt.accept(this);
      return hasStore;
    }
  };
  return FindStores(tensorVars, tempToBitGuard).hasStores(stmt);
}


Expr LowererImpl::searchForStartOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end) {
    taco_iassert(iterator.isWindowed());
    vector<Expr> args = {
            // Search over the `crd` array of the level,
            iterator.getMode().getModePack().getArray(1),
            // between the start and end position,
            start, end,
            // for the beginning of the window.
            iterator.getWindowLowerBound(),
    };
    return ir::Call::make("taco_binarySearchAfter", args, Datatype::UInt64);
}


Expr LowererImpl::searchForEndOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end) {
    taco_iassert(iterator.isWindowed());
    vector<Expr> args = {
            // Search over the `crd` array of the level,
            iterator.getMode().getModePack().getArray(1),
            // between the start and end position,
            start, end,
            // for the end of the window.
            iterator.getWindowUpperBound(),
    };
    return ir::Call::make("taco_binarySearchAfter", args, Datatype::UInt64);
}


Stmt LowererImpl::upperBoundGuardForWindowPosition(Iterator iterator, ir::Expr access) {
  taco_iassert(iterator.isWindowed());
  return ir::IfThenElse::make(
    ir::Gte::make(access, ir::Sub::make(iterator.getWindowUpperBound(), iterator.getWindowLowerBound())),
    ir::Break::make()
  );
}


Stmt LowererImpl::strideBoundsGuard(Iterator iterator, ir::Expr access, bool incrementPosVar) {
  Stmt cont = ir::Continue::make();
  // If requested to increment the iterator's position variable, add the increment
  // before the continue statement.
  if (incrementPosVar) {
    cont = ir::Block::make({
                               ir::Assign::make(iterator.getPosVar(),
                                                ir::Add::make(iterator.getPosVar(), ir::Literal::make(1))),
                               cont
                           });
  }
  // The guard makes sure that the coordinate being accessed is along the stride.
  return ir::IfThenElse::make(
      ir::Neq::make(ir::Rem::make(access, iterator.getStride()), ir::Literal::make(0)),
      cont
  );
}


Expr LowererImpl::projectWindowedPositionToCanonicalSpace(Iterator iterator, ir::Expr expr) {
  return ir::Div::make(ir::Sub::make(expr, iterator.getWindowLowerBound()), iterator.getStride());
}


Expr LowererImpl::projectCanonicalSpaceToWindowedPosition(Iterator iterator, ir::Expr expr) {
  return ir::Mul::make(ir::Add::make(expr, iterator.getWindowLowerBound()), iterator.getStride());
}

}
