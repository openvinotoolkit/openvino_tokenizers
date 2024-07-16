INDEX_NAME = res_index
IN_DIR = $(srcdir)
SRC_DIR = $(srcdir)

DIRS = $(TMP_DIR)/dirs.timestamp

$(TMP_DIR)/dirs.timestamp:
	$(MKINSTALLDIRS) $(OUT_DIR) $(TMP_DIR)
	echo timestamp > $(TMP_DIR)/dirs.timestamp

$(OUT_DIR)/calendar.res: $(IN_DIR)/calendar.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) calendar.txt

$(OUT_DIR)/casing.res: $(IN_DIR)/casing.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) casing.txt

$(OUT_DIR)/conversion.res: $(IN_DIR)/conversion.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) conversion.txt

$(OUT_DIR)/format.res: $(IN_DIR)/format.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) format.txt

$(OUT_DIR)/icuio.res: $(IN_DIR)/icuio.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) icuio.txt

$(OUT_DIR)/idna_rules.res: $(IN_DIR)/idna_rules.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) idna_rules.txt

$(OUT_DIR)/mc.res: $(IN_DIR)/mc.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) mc.txt

$(OUT_DIR)/root.res: $(IN_DIR)/root.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) root.txt

$(OUT_DIR)/sh_YU.res: $(IN_DIR)/sh_YU.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) sh_YU.txt

$(OUT_DIR)/sh.res: $(IN_DIR)/sh.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) sh.txt

$(OUT_DIR)/structLocale.res: $(IN_DIR)/structLocale.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) structLocale.txt

$(OUT_DIR)/te_IN_REVISED.res: $(IN_DIR)/te_IN_REVISED.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) te_IN_REVISED.txt

$(OUT_DIR)/te_IN.res: $(IN_DIR)/te_IN.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) te_IN.txt

$(OUT_DIR)/te.res: $(IN_DIR)/te.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) te.txt

$(OUT_DIR)/testaliases.res: $(IN_DIR)/testaliases.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) testaliases.txt

$(OUT_DIR)/testempty.res: $(IN_DIR)/testempty.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) testempty.txt

$(OUT_DIR)/testtypes.res: $(IN_DIR)/testtypes.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) testtypes.txt

$(OUT_DIR)/Thai_graphclust_model4_heavy.res: $(IN_DIR)/Thai_graphclust_model4_heavy.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) Thai_graphclust_model4_heavy.txt

$(OUT_DIR)/Thai_codepoints_exclusive_model5_heavy.res: $(IN_DIR)/Thai_codepoints_exclusive_model5_heavy.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) Thai_codepoints_exclusive_model5_heavy.txt

$(OUT_DIR)/Burmese_graphclust_model5_heavy.res: $(IN_DIR)/Burmese_graphclust_model5_heavy.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -q -s $(IN_DIR) -d $(OUT_DIR) Burmese_graphclust_model5_heavy.txt

$(OUT_DIR)/encoded.res: $(IN_DIR)/encoded.utf16be  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -s $(IN_DIR) -eUTF-16BE -d $(OUT_DIR) encoded.utf16be

$(TMP_DIR)/zoneinfo64.res: $(IN_DIR)/zoneinfo64.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -s $(IN_DIR) -d $(TMP_DIR) zoneinfo64.txt

$(OUT_DIR)/filtertest.res: $(IN_DIR)/filtertest.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -s $(IN_DIR) -d $(OUT_DIR) -i $(OUT_DIR) --filterDir $(IN_DIR)/filters filtertest.txt

$(OUT_DIR)/nfscsi.spp: $(IN_DIR)/nfs4_cs_prep_ci.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/gensprep -s $(IN_DIR) -d $(OUT_DIR) -b nfscsi -u 3.2.0 nfs4_cs_prep_ci.txt

$(OUT_DIR)/nfscss.spp: $(IN_DIR)/nfs4_cs_prep_cs.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/gensprep -s $(IN_DIR) -d $(OUT_DIR) -b nfscss -u 3.2.0 nfs4_cs_prep_cs.txt

$(OUT_DIR)/nfscis.spp: $(IN_DIR)/nfs4_cis_prep.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/gensprep -s $(IN_DIR) -d $(OUT_DIR) -b nfscis -u 3.2.0 -k -n $(IN_DIR)/../../data/unidata nfs4_cis_prep.txt

$(OUT_DIR)/nfsmxs.spp: $(IN_DIR)/nfs4_mixed_prep_s.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/gensprep -s $(IN_DIR) -d $(OUT_DIR) -b nfsmxs -u 3.2.0 -k -n $(IN_DIR)/../../data/unidata nfs4_mixed_prep_s.txt

$(OUT_DIR)/nfsmxp.spp: $(IN_DIR)/nfs4_mixed_prep_p.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/gensprep -s $(IN_DIR) -d $(OUT_DIR) -b nfsmxp -u 3.2.0 -k -n $(IN_DIR)/../../data/unidata nfs4_mixed_prep_p.txt

$(OUT_DIR)/test1.cnv: $(IN_DIR)/test1.ucm  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/makeconv --small -d $(OUT_DIR) $(IN_DIR)/test1.ucm

$(OUT_DIR)/test1bmp.cnv: $(IN_DIR)/test1bmp.ucm  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/makeconv --small -d $(OUT_DIR) $(IN_DIR)/test1bmp.ucm

$(OUT_DIR)/test2.cnv: $(IN_DIR)/test2.ucm  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/makeconv --small -d $(OUT_DIR) $(IN_DIR)/test2.ucm

$(OUT_DIR)/test3.cnv: $(IN_DIR)/test3.ucm  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/makeconv --small -d $(OUT_DIR) $(IN_DIR)/test3.ucm

$(OUT_DIR)/test4.cnv: $(IN_DIR)/test4.ucm  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/makeconv --small -d $(OUT_DIR) $(IN_DIR)/test4.ucm

$(OUT_DIR)/test4x.cnv: $(IN_DIR)/test4x.ucm  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/makeconv --small -d $(OUT_DIR) $(IN_DIR)/test4x.ucm

$(OUT_DIR)/test5.cnv: $(IN_DIR)/test5.ucm  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/makeconv --small -d $(OUT_DIR) $(IN_DIR)/test5.ucm

$(OUT_DIR)/ibm9027.cnv: $(IN_DIR)/ibm9027.ucm  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/makeconv --small -d $(OUT_DIR) $(IN_DIR)/ibm9027.ucm

$(OUT_DIR)/testnorm.nrm: $(IN_DIR)/testnorm.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/gennorm2 -s $(IN_DIR) testnorm.txt -o $(OUT_DIR)/testnorm.nrm

$(OUT_DIR)/test.icu:   | $(DIRS)
	$(INVOKE) $(GENTEST) -d $(OUT_DIR)

$(TMP_DIR)/testtable32.txt:   | $(DIRS)
	$(INVOKE) $(GENTEST) -r -d $(TMP_DIR)

$(OUT_DIR)/testtable32.res: $(TMP_DIR)/testtable32.txt  | $(DIRS)
	$(INVOKE) $(TOOLBINDIR)/genrb -s $(TMP_DIR) -d $(OUT_DIR) testtable32.txt

$(TMP_DIR)/nam.typ: $(OUT_DIR)/te.res  | $(DIRS)
	cp $(OUT_DIR)/te.res $(TMP_DIR)/nam.typ

$(OUT_DIR)/old_l_testtypes.res: $(IN_DIR)/old_l_testtypes.res  | $(DIRS)
	cp $(IN_DIR)/old_l_testtypes.res $(OUT_DIR)/old_l_testtypes.res

$(OUT_DIR)/old_e_testtypes.res: $(IN_DIR)/old_e_testtypes.res  | $(DIRS)
	cp $(IN_DIR)/old_e_testtypes.res $(OUT_DIR)/old_e_testtypes.res

define TESTDATA_LIST_CONTENT
Burmese_graphclust_model5_heavy.res
Thai_codepoints_exclusive_model5_heavy.res
Thai_graphclust_model4_heavy.res
calendar.res
casing.res
conversion.res
encoded.res
filtertest.res
format.res
ibm9027.cnv
icuio.res
idna_rules.res
mc.res
nfscis.spp
nfscsi.spp
nfscss.spp
nfsmxp.spp
nfsmxs.spp
old_e_testtypes.res
old_l_testtypes.res
root.res
sh.res
sh_YU.res
structLocale.res
te.res
te_IN.res
te_IN_REVISED.res
test.icu
test1.cnv
test1bmp.cnv
test2.cnv
test3.cnv
test4.cnv
test4x.cnv
test5.cnv
testaliases.res
testempty.res
testnorm.nrm
testtable32.res
testtypes.res
endef
export TESTDATA_LIST_CONTENT

$(TMP_DIR)/testdata.lst:   | $(DIRS)
	echo "$$TESTDATA_LIST_CONTENT" > $(TMP_DIR)/testdata.lst

TESTDATA_ALL_OUTPUT_FILES = $(OUT_DIR)/Burmese_graphclust_model5_heavy.res \
		$(OUT_DIR)/Thai_codepoints_exclusive_model5_heavy.res \
		$(OUT_DIR)/Thai_graphclust_model4_heavy.res \
		$(OUT_DIR)/calendar.res \
		$(OUT_DIR)/casing.res \
		$(OUT_DIR)/conversion.res \
		$(OUT_DIR)/encoded.res \
		$(OUT_DIR)/filtertest.res \
		$(OUT_DIR)/format.res \
		$(OUT_DIR)/ibm9027.cnv \
		$(OUT_DIR)/icuio.res \
		$(OUT_DIR)/idna_rules.res \
		$(OUT_DIR)/mc.res \
		$(TMP_DIR)/nam.typ \
		$(OUT_DIR)/nfscis.spp \
		$(OUT_DIR)/nfscsi.spp \
		$(OUT_DIR)/nfscss.spp \
		$(OUT_DIR)/nfsmxp.spp \
		$(OUT_DIR)/nfsmxs.spp \
		$(OUT_DIR)/old_e_testtypes.res \
		$(OUT_DIR)/old_l_testtypes.res \
		$(OUT_DIR)/root.res \
		$(OUT_DIR)/sh.res \
		$(OUT_DIR)/sh_YU.res \
		$(OUT_DIR)/structLocale.res \
		$(OUT_DIR)/te.res \
		$(OUT_DIR)/te_IN.res \
		$(OUT_DIR)/te_IN_REVISED.res \
		$(OUT_DIR)/test.icu \
		$(OUT_DIR)/test1.cnv \
		$(OUT_DIR)/test1bmp.cnv \
		$(OUT_DIR)/test2.cnv \
		$(OUT_DIR)/test3.cnv \
		$(OUT_DIR)/test4.cnv \
		$(OUT_DIR)/test4x.cnv \
		$(OUT_DIR)/test5.cnv \
		$(OUT_DIR)/testaliases.res \
		$(TMP_DIR)/testdata.lst \
		$(OUT_DIR)/testempty.res \
		$(OUT_DIR)/testnorm.nrm \
		$(OUT_DIR)/testtable32.res \
		$(TMP_DIR)/testtable32.txt \
		$(OUT_DIR)/testtypes.res \
		$(TMP_DIR)/zoneinfo64.res


