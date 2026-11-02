#include "RubbleTypes.h"

const TCHAR* LexToString(ERubbleFamily Family)
{
	switch (Family)
	{
	case ERubbleFamily::Rigid: return TEXT("rigid");
	case ERubbleFamily::Vehicles: return TEXT("vehicles");
	case ERubbleFamily::Soft: return TEXT("soft");
	case ERubbleFamily::Fracture: return TEXT("fracture");
	}
	return TEXT("?");
}

bool LexFromString(ERubbleFamily& Out, const FString& Text)
{
	static const TPair<const TCHAR*, ERubbleFamily> Names[] = {
		{ TEXT("rigid"), ERubbleFamily::Rigid },
		{ TEXT("vehicles"), ERubbleFamily::Vehicles },
		{ TEXT("vehicle"), ERubbleFamily::Vehicles },
		{ TEXT("soft"), ERubbleFamily::Soft },
		{ TEXT("softbody"), ERubbleFamily::Soft },
		{ TEXT("fracture"), ERubbleFamily::Fracture },
	};
	for (const auto& N : Names)
	{
		if (Text.Equals(N.Key, ESearchCase::IgnoreCase))
		{
			Out = N.Value;
			return true;
		}
	}
	return false;
}

bool FRubbleSceneRequest::Parse(const FString& Text, FRubbleSceneRequest& Out, FString& Error)
{
	// family:id?key=value&key=value
	FString FamilyText, Rest;
	if (!Text.Split(TEXT(":"), &FamilyText, &Rest))
	{
		Error = FString::Printf(TEXT("'%s' is not family:scene (rigid, vehicles, soft, fracture)"), *Text);
		return false;
	}
	FRubbleSceneRequest R;
	if (!LexFromString(R.Family, FamilyText))
	{
		Error = FString::Printf(TEXT("unknown family '%s' (rigid, vehicles, soft, fracture)"), *FamilyText);
		return false;
	}
	FString Query;
	if (!Rest.Split(TEXT("?"), &R.Id, &Query))
	{
		R.Id = Rest;
	}
	if (R.Id.IsEmpty())
	{
		Error = TEXT("no scene id");
		return false;
	}
	TArray<FString> Pairs;
	Query.ParseIntoArray(Pairs, TEXT("&"));
	for (const FString& Pair : Pairs)
	{
		FString Key, Value;
		if (!Pair.Split(TEXT("="), &Key, &Value) || Key.IsEmpty() || !Value.IsNumeric())
		{
			Error = FString::Printf(TEXT("parameter '%s' is not key=number"), *Pair);
			return false;
		}
		R.Params.Add(Key, FCString::Atof(*Value));
	}
	Out = MoveTemp(R);
	return true;
}

FString FRubbleSceneRequest::ToString() const
{
	FString S = FString::Printf(TEXT("%s:%s"), LexToString(Family), *Id);
	TCHAR Sep = TEXT('?');
	for (const TPair<FString, float>& P : Params)
	{
		S += FString::Printf(TEXT("%c%s=%g"), Sep, *P.Key, P.Value);
		Sep = TEXT('&');
	}
	return S;
}
