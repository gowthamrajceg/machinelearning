
import numpy as np
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
# AND data
ANDtargets = np.array([[0],[0],[0],[1]])
# OR data
ORtargets = np.array([[0],[1],[1],[1]])
# XOR data
XORtargets = np.array([[0],[1],[1],[0]])
import pcn_logic_eg

print "AND logic function"
pAND = pcn_logic_eg.pcn(inputs,ANDtargets)
pAND.pcntrain(inputs,ANDtargets,0.25,6) #giving eta the value .4 and iterations 4.
print "************************************************************************************************************************************"
print "OR logic function"
pOR = pcn_logic_eg.pcn(inputs,ORtargets)
pOR.pcntrain(inputs,ORtargets,0.25,6)
print "************************************************************************************************************************************"
print "XOR logic function"
pXOR = pcn_logic_eg.pcn(inputs,XORtargets)
pXOR.pcntrain(inputs,XORtargets,0.25,6)
print "************************************************************************************************************************************"
print "Training rate eta is 0.01 and iterations is 10 for AND"
pANDMod = pcn_logic_eg.pcn(inputs,ANDtargets)
pANDMod.pcntrain(inputs,ANDtargets,0.01,10)
print "************************************************************************************************************************************"
print "Training rate eta is 0.5 and iterations is 6 for AND"
pANDMod = pcn_logic_eg.pcn(inputs,ANDtargets)
pANDMod.pcntrain(inputs,ANDtargets,0.5,6)
print "************************************************************************************************************************************"
print "XOR with eta value as 0.4 and 10 iterations"
pXORMod= pcn_logic_eg.pcn(inputs,XORtargets)
pXORMod.pcntrain(inputs,XORtargets,0.4,10)
print "Still not correct!"
